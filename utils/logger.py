"""
Logging utility for SRGAN training and evaluation
Provides structured logging with timestamps and file output
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from utils.config import Config

def setup_logger(name: str = "SRGAN", log_file: str = None) -> logging.Logger:
    """
    Setup logger with both file and console handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path (default: logs/srgan_YYYYMMDD_HHMMSS.log)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Config.LOGS_DIR / f"srgan_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
