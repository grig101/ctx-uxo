"""
Device management utilities for the UXO detection pipeline.

This module provides utilities for detecting and managing hardware devices,
including GPU detection, CUDA validation, and device selection.
"""

import torch
import os
import sys
from typing import Optional, Dict, Any


class DeviceManager:
    """
    Manages hardware device detection and configuration.
    
    This class provides comprehensive device management including
    GPU detection, CUDA validation, and automatic device selection.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the device manager.
        
        Args:
            logger: Logger instance for output messages
        """
        self.logger = logger
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information.
        
        Returns:
            Dictionary containing device information
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'memory_total': None,
            'memory_allocated': None,
            'cuda_version': None,
            'pytorch_version': torch.__version__,
            'python_version': sys.version
        }
        
        if info['cuda_available']:
            info['current_device'] = torch.cuda.current_device()
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
            info['memory_allocated'] = torch.cuda.memory_allocated(0)
            info['cuda_version'] = torch.version.cuda
        
        return info
    
    def log_device_info(self) -> None:
        """Log comprehensive device information."""
        if self.logger is None:
            return
        
        self.logger.info(f"PyTorch Version: {self.device_info['pytorch_version']}")
        self.logger.info(f"Python Version: {self.device_info['python_version']}")
        
        if self.device_info['cuda_available']:
            self.logger.info("CUDA is available")
            self.logger.info(f"GPU Count: {self.device_info['device_count']}")
            self.logger.info(f"GPU Name: {self.device_info['device_name']}")
            
            memory_gb = self.device_info['memory_total'] / (1024**3)
            self.logger.info(f"Total GPU Memory: {memory_gb:.2f} GB")
            
            if memory_gb >= 15:
                self.logger.info("GPU memory sufficient for training")
            else:
                self.logger.warning("GPU memory may be insufficient for training")
            
            if self.device_info['cuda_version']:
                cuda_major = int(self.device_info['cuda_version'].split('.')[0])
                if cuda_major >= 12:
                    self.logger.info(f"CUDA Version: {self.device_info['cuda_version']} (Compatible)")
                else:
                    self.logger.warning(f"CUDA Version: {self.device_info['cuda_version']} (Recommended: 12+)")
        else:
            self.logger.warning("CUDA is not available - using CPU")
    
    def get_optimal_device(self, preferred_device: str = "auto") -> torch.device:
        """
        Get the optimal device for computation.
        
        Args:
            preferred_device: Preferred device ("auto", "cuda", "cpu")
            
        Returns:
            torch.device: Optimal device
        """
        if preferred_device == "auto":
            if torch.cuda.is_available():
                optimal_device = torch.device("cuda")
                if self.logger:
                    self.logger.info(f"Optimal device: {optimal_device}")
                else:
                    print(f"Optimal device: {optimal_device}")
            else:
                optimal_device = torch.device("cpu")
                if self.logger:
                    self.logger.info(f"Optimal device: {optimal_device}")
                else:
                    print(f"Optimal device: {optimal_device}")
        else:
            optimal_device = torch.device(preferred_device)
        
        memory_usage = self.get_memory_usage()
        if self.logger:
            self.logger.info(f"Memory usage: {memory_usage}")
        else:
            print(f"Memory usage: {memory_usage}")
        
        return optimal_device
    
    def validate_device_requirements(self, min_memory_gb: float = 15.0) -> bool:
        """
        Validate device meets minimum requirements.
        
        Args:
            min_memory_gb: Minimum GPU memory in GB
            
        Returns:
            True if device meets requirements, False otherwise
        """
        if not self.device_info['cuda_available']:
            return False
        
        memory_gb = self.device_info['memory_total'] / (1024**3)
        return memory_gb >= min_memory_gb
    
    def setup_device_optimizations(self, device: torch.device) -> None:
        """
        Setup device-specific optimizations.
        
        Args:
            device: Target device for optimization
        """
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            if self.logger:
                self.logger.info("CUDA optimizations applied")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage in GB
        """
        memory_info = {
            'gpu_total': 0.0,
            'gpu_allocated': 0.0,
            'gpu_free': 0.0
        }
        
        if self.device_info['cuda_available']:
            memory_info['gpu_total'] = self.device_info['memory_total'] / (1024**3)
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_free'] = memory_info['gpu_total'] - memory_info['gpu_allocated']
        
        return memory_info


def get_device_info(logger=None) -> bool:
    """
    Legacy function for device information logging.
    
    This function maintains backward compatibility while using
    the new DeviceManager class internally.
    
    Args:
        logger: Logger instance for output
        
    Returns:
        True if CUDA is available and meets requirements, False otherwise
    """
    device_manager = DeviceManager(logger)
    device_manager.log_device_info()
    
    return device_manager.device_info['cuda_available'] and device_manager.validate_device_requirements()


def get_optimal_device(preferred_device: str = "auto") -> torch.device:
    """
    Get optimal device for computation.
    
    Args:
        preferred_device: Preferred device ('auto', 'cuda', 'cpu')
        
    Returns:
        torch.device: Optimal device for computation
    """
    device_manager = DeviceManager()
    return device_manager.get_optimal_device(preferred_device)


if __name__ == "__main__":
    device_manager = DeviceManager()
    device_manager.log_device_info()
    
    optimal_device = device_manager.get_optimal_device()
    print(f"Optimal device: {optimal_device}")
    
    memory_usage = device_manager.get_memory_usage()
    print(f"Memory usage: {memory_usage}")