import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, log_dir="./logs", log_name="logs", date_str="./", max_bytes=10*1024*1024, backup_count=5):
        """
        Enhanced logger with rotation and better error handling.
        
        Args:
            log_dir (str): Directory for log files
            log_name (str): Base name for log files
            date_str (str): Date string for log organization
            max_bytes (int): Maximum size of log file before rotation
            backup_count (int): Number of backup files to keep
        """
        log_path = os.path.join(log_dir, f"{date_str}/{log_name}.log")
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.logger.handlers.clear()

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

        self.file_handler = file_handler
        self.stream_handler = stream_handler

    def info(self, message):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)

    def critical(self, message):
        """Log critical message."""
        self.logger.critical(message)

    def exception(self, message):
        """Log exception with traceback."""
        self.logger.exception(message)

    def close(self):
        """Properly close all handlers."""
        try:
            self.logger.removeHandler(self.file_handler)
            self.logger.removeHandler(self.stream_handler)
            self.file_handler.close()
            self.stream_handler.close()
        except Exception as e:
            print(f"Error closing logger: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
