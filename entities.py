# File: entities.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PIIEntity:
    """Data class for PII entities"""
    value: str
    confidence: float
    model_source: str
    category: str = "UNKNOWN"

# File: patterns.py
import re
from typing import Dict, Set, Pattern

class PatternManager:
    def __init__(self):
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
        
        self.compiled_patterns: Dict[str, Pattern] = {
            name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for name, pattern in self.patterns.items()
        }
        
        self.false_positives: Set[str] = frozenset({
            'DEBUG', 'INFO', 'ERROR', 'WARNING', 'null',
            'GET', 'POST', 'PUT', 'DELETE', 'PATCH',
            'true', 'false', 'None', 'undefined'
        })

    def clean_value(self, value: str) -> str:
        """Clean detected values with improved handling"""
        if not value or len(value) < 2:
            return ""
            
        value = value.strip()
        if value.upper() in self.false_positives:
            return ""
            
        value = re.sub(r'^\W+|\W+$', '', value)
        value = ' '.join(value.split())
        
        return value if len(value) >= 2 else ""