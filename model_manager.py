# File: model_manager.py
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging
from typing import Optional, Dict, List
from entities import PIIEntity

class ModelManager:
    def __init__(self, batch_size: int, device: torch.device):
        self.device = device
        self.batch_size = batch_size
        self.confidence_threshold = 0.75
        self.use_amp = device.type == "cuda"
        self._initialize_models()

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
            
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise

    def map_ner_labels(self, ner_result: List[Dict]) -> List[PIIEntity]:
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
                value = item['word'].strip()
                
                if value:
                    entities.append(PIIEntity(
                        value=value,
                        confidence=item['score'],
                        model_source="bert-ner",
                        category=category
                    ))
        
        return entities

    def cleanup(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None

# File: text_processor.py
from typing import List

class TextProcessor:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length

    def split_text(self, text: str) -> List[str]:
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