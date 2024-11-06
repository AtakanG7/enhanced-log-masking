import time
import torch
import re
import psutil
from typing import List, Set, Dict, Optional, Tuple, Iterator, Type, Union, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import functools
import gc
import logging
import contextlib
from pathlib import Path
from types import TracebackType
from monitor import ResourceMonitor
from dataclasses import dataclass

@dataclass(frozen=True)
class PIIEntity:
    """Data class for PII entities"""
    value: str
    confidence: float
    model_source: str
    category: str = "UNKNOWN"

class DynamicEnsemblePIIDetector:
    """Dynamic ensemble PII detector combining pattern matching and NLP"""
    def __init__(self, initial_batch_size: int = 16, max_workers: Optional[int] = None):
        """Initialize detector with dynamic batch sizing"""
        self.batch_size = initial_batch_size
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 2) + 4)
        self.max_length = 512
        self.confidence_threshold = 0.75
        self.resource_monitor = ResourceMonitor()
        self.performance_stats = {
            'processed_batches': 0,
            'total_processing_time': 0,
            'batch_size_adjustments': 0
        }
        self._initialize_patterns()
        self._initialize_system()

    def __enter__(self) -> 'DynamicEnsemblePIIDetector':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]) -> None:
        try:
            self.cleanup()
        except Exception as e:
            logging.error(f"Error during context manager cleanup: {e}")
        return False

    def _initialize_patterns(self):
        """Initialize regex patterns for PII detection"""
        self.patterns = {
            'CREDIT_CARD': r'\b\d{4}(?:[- ]?\d{4}){3}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'EMAIL': r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b',
            'IP_ADDRESS': r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            'PHONE': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'DATE': r'\b\d{4}[-/]\d{2}[-/]\d{2}\b',
            'MAC_ADDRESS': r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
            'IBAN': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
        }
        
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for name, pattern in self.patterns.items()
        }
        
        self.false_positives = frozenset({
            'DEBUG', 'INFO', 'ERROR', 'WARNING', 'null',
            'GET', 'POST', 'PUT', 'DELETE', 'PATCH',
            'true', 'false', 'None', 'undefined'
        })

    def _clean_value(self, value: str) -> str:
        """Clean detected values with improved handling"""
        if not value or len(value) < 2:
            return ""
            
        value = value.strip()
        if value.upper() in self.false_positives:
            return ""
            
        value = re.sub(r'^\W+|\W+$', '', value)
        value = ' '.join(value.split())
        
        return value if len(value) >= 2 else ""

    def _initialize_system(self):
        """Initialize system with resource monitoring"""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_available = gpu_memory - torch.cuda.memory_allocated()
                
                if gpu_memory_available > 2 * 1024 * 1024 * 1024:
                    self.device = torch.device("cuda")
                    self.use_amp = True
                else:
                    logging.warning("Insufficient GPU memory. Falling back to CPU.")
                    self.device = torch.device("cpu")
                    self.use_amp = False
            else:
                self.device = torch.device("cpu")
                self.use_amp = False
            
            self._initialize_models()
            
            cache_size = min(
                2048,
                int(psutil.virtual_memory().available / (1024 * 1024 * 10))
            )
            self.clean_value = functools.lru_cache(maxsize=cache_size)(self._clean_value)
            
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            raise

    def _initialize_models(self):
        """Initialize NLP models for PII detection"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
            self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
            
            self.model.to(self.device)
            
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                batch_size=self.batch_size,
                aggregation_strategy="simple"
            )
            
            self.model_max_length = self.tokenizer.model_max_length
            
            if self.device.type == "cuda":
                self.model.gradient_checkpointing_enable()
            
            self.model.eval()
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise

    def detect_with_patterns(self, text: str) -> List[PIIEntity]:
        """Detect PII using regex patterns"""
        entities = []
        
        for category, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                value = self.clean_value(match.group())
                if value:
                    entities.append(PIIEntity(
                        value=value,
                        confidence=1.0,
                        model_source="pattern",
                        category=category
                    ))
        
        return entities

    def _map_ner_labels(self, ner_result: List[Dict]) -> List[PIIEntity]:
        """Map NER labels to PII entities"""
        entities = []
        label_mapping = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'MISC': 'MISCELLANEOUS'
        }
        
        for item in ner_result:
            if item['score'] >= self.confidence_threshold:
                category = label_mapping.get(item['entity_group'], 'UNKNOWN')
                value = self.clean_value(item['word'])
                
                if value:
                    entities.append(PIIEntity(
                        value=value,
                        confidence=item['score'],
                        model_source="bert-ner",
                        category=category
                    ))
        
        return entities

    def process_batch(self, texts: List[str]) -> List[List[PIIEntity]]:
        """Process a batch of texts for PII detection"""
        start_time = time.time()
        results = []
        
        try:
            pattern_entities = [self.detect_with_patterns(text) for text in texts]
            
            with torch.cuda.amp.autocast() if self.use_amp and self.device.type == "cuda" else contextlib.nullcontext():
                ner_results = self.ner_pipeline(texts)
            
            ner_entities = [self._map_ner_labels(result) if isinstance(result, list) else [] 
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
            
            if should_adjust:
                self.batch_size = new_batch_size
                self.performance_stats['batch_size_adjustments'] += 1
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            return [[] for _ in texts]

    def process_text_chunk(self, text: str) -> List[PIIEntity]:

        """Process a single chunk of text"""
        if not text.strip():
            return []

        try:
            chunks = self._split_text(text)
            all_entities = []
            
            for chunk in chunks:
                entities = self.process_batch([chunk])[0]
                all_entities.extend(entities)
            
            return self._deduplicate_entities(all_entities)
            
        except Exception as e:
            logging.error

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks that respect word boundaries"""
        if len(text) <= self.max_length:
            return [text]
        
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            chunk_end = min(current_pos + self.max_length, text_length)
            
            if chunk_end < text_length:
                last_space = text.rfind(' ', current_pos, chunk_end)
                if last_space != -1:
                    chunk_end = last_space
            
            chunk = text[current_pos:chunk_end].strip()
            if chunk:
                chunks.append(chunk)
            
            current_pos = chunk_end + 1
        
        return chunks

    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Deduplicate PII entities while preserving the highest confidence ones"""
        unique_entities: Dict[Tuple[str, str], PIIEntity] = {}
        
        for entity in entities:
            key = (entity.value, entity.category)
            if key not in unique_entities or entity.confidence > unique_entities[key].confidence:
                unique_entities[key] = entity
        
        return list(unique_entities.values())

    def process_file_stream(self, file_path: Union[str, Path], chunk_size: int = 1024*1024) -> Iterator[List[PIIEntity]]:
        """Process a file in streaming fashion, yielding PII entities as they're found"""
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
                    entities = self.process_text_chunk(text_chunk)
                    if entities:
                        yield entities
                        
        except Exception as e:
            logging.error(f"Error processing file stream: {e}")
            raise

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        stats = self.performance_stats.copy()
        if stats['processed_batches'] > 0:
            stats['average_processing_time'] = (
                stats['total_processing_time'] / stats['processed_batches']
            )
        return stats

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Clear CUDA cache if using GPU
            if hasattr(self, 'device') and self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Clear model from memory
            if hasattr(self, 'model'):
                self.model = None
            if hasattr(self, 'tokenizer'):
                self.tokenizer = None
            if hasattr(self, 'ner_pipeline'):
                self.ner_pipeline = None
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            raise
