from dataclasses import dataclass
import logging
import argparse
from scale import DynamicEnsemblePIIDetector

@dataclass(frozen=True)
class PIIEntity:
    """Data class for PII entities"""
    value: str
    confidence: float
    model_source: str
    category: str = "UNKNOWN"

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='PII Detection in Text Files')
    parser.add_argument('input_file', help='Path to input file')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Initial batch size for processing')
    parser.add_argument('--output-file', help='Path to output file (optional)')
    parser.add_argument('--log-level', default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Set logging level')
    parser.add_argument('--chunk-size', type=int, default=1024*1024,
                      help='Size of chunks to process (in bytes)')
    parser.add_argument('--confidence-threshold', type=float, default=0.75,
                      help='Minimum confidence threshold for NER detection')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        detector = DynamicEnsemblePIIDetector(
            initial_batch_size=args.batch_size
        )
        
        with detector:
            output_file = args.output_file or f"{args.input_file}.pii_detected"
            stats_file = f"{output_file}.stats"
            
            with open(output_file, 'w', encoding='utf-8') as out_f:
                out_f.write("Category\tValue\tConfidence\tSource\n")  # Header
                
                # Process file in streaming fashion
                for entities in detector.process_file_stream(args.input_file, chunk_size=args.chunk_size):
                    for entity in entities:
                        out_f.write(f"{entity.category}\t{entity.value}\t"
                                  f"{entity.confidence:.4f}\t{entity.model_source}\n")
            
            # Write performance statistics
            stats = detector.get_performance_stats()
            with open(stats_file, 'w', encoding='utf-8') as stats_f:
                stats_f.write("Performance Statistics:\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        stats_f.write(f"{key}: {value:.4f}\n")
                    else:
                        stats_f.write(f"{key}: {value}\n")
                    logging.info(f"{key}: {value}")
                
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()