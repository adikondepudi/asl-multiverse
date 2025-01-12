import logging
from pathlib import Path
import sys
from typing import Optional
import json
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'metrics'):
            log_data['metrics'] = record.metrics
            
        return json.dumps(log_data)

def setup_logging(log_dir: str = 'logs',
                 level: str = 'INFO',
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 json_logging: bool = False) -> None:
    """
    Setup logging configuration
    
    Parameters
    ----------
    log_dir : str
        Directory for log files
    level : str
        Logging level
    log_to_file : bool
        Whether to log to file
    log_to_console : bool
        Whether to log to console
    json_logging : bool
        Whether to use JSON formatting
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    handlers = []
    
    # File handler
    if log_to_file:
        log_file = log_dir / f'asl_training_{datetime.now():%Y%m%d_%H%M%S}.log'
        file_handler = logging.FileHandler(log_file)
        
        if json_logging:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        handlers.append(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        handlers.append(console_handler)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        logger.addHandler(handler)

class MetricLogger:
    """Logger for training metrics"""
    
    def __init__(self, logger_name: str = 'metrics'):
        self.logger = logging.getLogger(logger_name)
        self.step = 0
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics"""
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg=f'Metrics at step {self.step}',
            args=(),
            exc_info=None
        )
        record.metrics = metrics
        
        self.logger.handle(record)

# Example usage
if __name__ == '__main__':
    # Setup logging
    setup_logging(level='DEBUG', json_logging=True)
    
    # Create metric logger
    metric_logger = MetricLogger()
    
    # Log some metrics
    metrics = {
        'loss': 0.5,
        'accuracy': 0.95
    }
    metric_logger.log_metrics(metrics)