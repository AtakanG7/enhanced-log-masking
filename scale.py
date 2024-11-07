import time
import torch
import psutil
import gc
import logging
import contextlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Type
from types import TracebackType
import functools

from entities import PIIEntity
from entities import PatternManager
from model_manager import ModelManager
from model_manager import TextProcessor
from monitor import ResourceMonitor

class DynamicEnsemblePIIDetector:
    def __init__(self, initial_batch_size: int = 16, max_workers: Optional[int] = None):
        self.batch_size = initial_batch_size
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 2) + 4)
        self.max_length = 512
        self.resource_monitor = ResourceMonitor()
        self.performance_stats = {
            'processed_batches': 0,
            'total_processing_time': 0,
            'batch_size_adjustments': 0
        }
        self._initialize_system()
        
    def _initialize_system(self):
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_available = gpu_memory - torch.cuda.memory_allocated()
                
                if gpu_memory_available > 2 * 1024 * 1024 * 1024:
                    self.device = torch.device("cuda")
                else:
                    logging.warning("Insufficient GPU memory. Falling back to CPU.")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
            
            self.pattern_manager = PatternManager()
            self.model_manager = ModelManager(self.batch_size, self.device)
            self.text_processor = TextProcessor(self.max_length)
            
            cache_size = min(
                2048,
                int(psutil.virtual_memory().available / (1024 * 1024 * 10))
            )
            self.pattern_manager.clean_value = functools.lru_cache(maxsize=cache_size)(
                self.pattern_manager.clean_value
            )
            
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]) -> None:
        try:
            self.cleanup()
        except Exception as e:
            logging.error(f"Error during context manager cleanup: {e}")
        return False

    def detect_with_patterns(self, text: str) -> List[PIIEntity]:
        entities = []
        for category, pattern in self.pattern_manager.compiled_patterns.items():
            for match in pattern.finditer(text):
                value = self.pattern_manager.clean_value(match.group())
                if value:
                    entities.append(PIIEntity(
                        value=value,
                        confidence=1.0,
                        model_source="pattern",
                        category=category
                    ))
        return entities

    def process_batch(self, texts: List[str]) -> List[List[PIIEntity]]:
        start_time = time.time()
        results = []
        
        try:
            pattern_entities = [self.detect_with_patterns(text) for text in texts]
            with torch.cuda.amp.autocast() if self.model_manager.use_amp else contextlib.nullcontext():
                print(f"Running NER on {texts} texts")
                ner_results = self.model_manager.ner_pipeline(texts)
            
            ner_entities = [self.model_manager.map_ner_labels(result) if isinstance(result, list) else [] 
                           for result in ner_results]
            
            for pattern_ents, ner_ents in zip(pattern_entities, ner_entities):
                combined = pattern_ents + ner_ents
                seen = set()
                unique_entities = []
                for entity in combined:
                    key = (entity.value, entity.category)
                    if key not in seen:
                        seen.add(key)
                        unique_entities.append(entity)
                results.append(unique_entities)
            
            processing_time = time.time() - start_time
            self.performance_stats['processed_batches'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            new_batch_size, should_adjust = self.resource_monitor.calculate_optimal_batch_size(
                self.batch_size, processing_time
            )
            print(f"Batch size: {self.batch_size}, new batch size: {new_batch_size}, should adjust: {should_adjust}")
            if should_adjust:
                self.batch_size = new_batch_size
                self.performance_stats['batch_size_adjustments'] += 1
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            return [[] for _ in texts]

    def process_text_chunk(self, text: str) -> List[PIIEntity]:
        if not text.strip():
            return []

        try:
            chunks = self.text_processor.split_text(text)
            all_entities = []
            
            for chunk in chunks:
                print(f"Processing chunk: {chunk}")
                entities = self.process_batch([chunk])[0]
                all_entities.extend(entities)
            
            return self._deduplicate_entities(all_entities)
            
        except Exception as e:
            logging.error(f"Error processing text chunk: {e}")
            return []

    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        unique_entities: Dict[tuple, PIIEntity] = {}
        
        for entity in entities:
            key = (entity.value, entity.category)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        return list(unique_entities.values())

    def process_file_stream(self, file_path: Union[str, Path], chunk_size: int = 1024*1024):
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                buffer = []
                current_size = 0
                
                for line in f:
                    buffer.append(line)
                    current_size += len(line.encode('utf-8'))
                    
                    if current_size >= chunk_size:
                        text_chunk = ''.join(buffer)
                        entities = self.process_text_chunk(text_chunk)
                        if entities:
                            yield entities
                        
                        buffer = []
                        current_size = 0
                
                if buffer:
                    text_chunk = ''.join(buffer)
                    print(f"Remaining text chunk: {text_chunk}")
                    entities = self.process_text_chunk(text_chunk)
                    if entities:
                        yield entities
                        
        except Exception as e:
            logging.error(f"Error processing file stream: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, float]:
        stats = self.performance_stats.copy()
        if stats['processed_batches'] > 0:
            stats['average_processing_time'] = (
                stats['total_processing_time'] / stats['processed_batches']
            )
        return stats

    def cleanup(self) -> None:
        try:
            self.model_manager.cleanup()
            gc.collect()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            raise