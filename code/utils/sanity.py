"""
Sanity check utilities for the UXO detection pipeline.

This module provides utilities for validating the environment and creating
necessary directories for the pipeline execution.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_environment() -> bool:
    """
    Validate the execution environment.
    
    Returns:
        bool: True if environment is valid
    """
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ is required")
        return False
    
    required_dirs = ["./dataset", "./logs", "./results"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Warning: Required directory '{dir_name}' not found")
            return False
    
    return True


def sanity_check(log_dir: str = "./logs", date_str: str = "./") -> None:
    """
    Perform comprehensive sanity checks and setup.
    
    This function validates the environment, creates necessary directories,
    and ensures the pipeline is ready for execution.
    
    Args:
        log_dir: Directory for log files
        date_str: Date string for organizing logs
        
    Raises:
        RuntimeError: If critical setup fails
    """
    if not validate_environment():
        raise RuntimeError("Environment validation failed")
    
    directories = [
        log_dir,
        f"{log_dir}/{date_str}",
        "./results",
        f"./results/{date_str}",
        "./debug",
        "./dataset/backgrounds"
    ]
    
    for directory in directories:
        create_directory(directory)


def check_model_files(model_path: str) -> bool:
    """
    Check if model files exist and are accessible.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        True if model files are valid, False otherwise
    """
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    if file_size < 1024:
        print(f"Warning: Model file seems too small: {file_size} bytes")
        return False
    
    return True


def check_dataset_structure() -> bool:
    """
    Validate dataset directory structure.
    
    Returns:
        True if dataset structure is valid, False otherwise
    """
    dataset_path = Path("./dataset")
    
    if not dataset_path.exists():
        print("Warning: Dataset directory not found")
        return False
    
    expected_dirs = ['train', 'val', 'test']
    for subdir in expected_dirs:
        if not (dataset_path / subdir).exists():
            print(f"Warning: Dataset subdirectory '{subdir}' not found")
    
    return True


def setup_experiment_environment(experiment_name: Optional[str] = None) -> str:
    """
    Setup environment for a new experiment.
    
    Args:
        experiment_name: Optional name for the experiment
        
    Returns:
        Experiment directory path
    """
    from datetime import datetime
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = f"./experiments/{experiment_name}"
    create_directory(experiment_dir)
    
    subdirs = ['logs', 'checkpoints', 'results', 'configs']
    for subdir in subdirs:
        create_directory(f"{experiment_dir}/{subdir}")
    
    return experiment_dir



