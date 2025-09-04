"""
UXO Detection Model Pipeline CLI.

This module provides a command-line interface for training, testing, and deploying
the UXO (Unexploded Ordnance) detection model using Faster R-CNN architecture.
Another architectures/models will be added in time.
"""

import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from utils.sanity import sanity_check
from utils.logger import Logger
from tools.train import train_model
from tools.test import test_model
from deploy.runAPI import run_app


class ConfigManager:
    """
    Manages configuration loading and argument parsing with config.yaml defaults.
    
    This class handles the loading of configuration files and provides
    a unified interface for accessing configuration parameters across
    the entire application.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file. If None, 
                        defaults to 'config.yaml' in the current directory.
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration parameters.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is malformed.
        """
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
                key: Configuration key (e.g., 'training.num_epochs')
                default: Default value if key doesn't exist

        Returns:
                Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


class ArgumentParser:
    """
    Handles command-line argument parsing with config.yaml integration.
    
    This class creates and manages argument parsers for different
    subcommands while ensuring proper integration with configuration
    file defaults.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the argument parser.

        Args:
        config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.parser = self._create_main_parser()
        self.subparsers = self._create_subparsers()
    
    def _create_main_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        return argparse.ArgumentParser(
            description="UXO Detection Model Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog= """
                    Examples:
                    python main.py train --config config.yaml OR specify parameters via CLI
                    python main.py test --model_path ./results/best_model.pth
                    python main.py deploy --device auto
                    """
                    )
    
    def _create_subparsers(self) -> argparse._SubParsersAction:
        """Create subparsers for different commands."""
        subparsers = self.parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        self._add_train_parser(subparsers)
        self._add_test_parser(subparsers)
        self._add_deploy_parser(subparsers)
        
        return subparsers
    
    def _add_train_parser(self, subparsers: argparse._SubParsersAction):
        """Add training subcommand parser."""
        train_parser = subparsers.add_parser(
            'train',
            help='Train the UXO detection model',
            description='Fine-tune the Faster R-CNN model on CTX UXO dataset'
        )
        
        train_parser.add_argument(
            '--backbone',
            type=str,
            default=self.config_manager.get_value('model.backbone', 'resnet50'),
            help='Backbone architecture (resnet18, resnet50, mobilenet)'
        )
        train_parser.add_argument(
            '--input_size',
            type=int,
            default=self.config_manager.get_value('model.input_size', 800),
            help='Input image size (square)'
        )
        
        train_parser.add_argument(
            '--num_epochs',
            type=int,
            default=self.config_manager.get_value('training.num_epochs', 300),
            help='Number of training epochs'
        )

        train_parser.add_argument(
            '--batch_size',
            type=int,
            default=self.config_manager.get_value('training.batch_size', 16),
            help='Training batch size'
        )

        train_parser.add_argument(
            '--test_batch_size',
            type=int,
            default=self.config_manager.get_value('training.test_batch_size', 1),
            help='Testing batch size'
        )

        train_parser.add_argument(
            '--learning_rate',
            type=float,
            default=self.config_manager.get_value('training.learning_rate', 0.005),
            help='Learning rate'
        )

        train_parser.add_argument(
            '--weight_decay',
            type=float,
            default=self.config_manager.get_value('training.weight_decay', 0.0001),
            help='Weight decay'
        )

        train_parser.add_argument(
            '--momentum',
            type=float,
            default=self.config_manager.get_value('training.momentum', 0.9),
            help='SGD momentum'
        )

        train_parser.add_argument(
            '--early_stopping_epochs',
            type=int,
            default=self.config_manager.get_value('training.early_stopping_epochs', 50),
            help='Early stopping patience'
        )

        train_parser.add_argument(
            '--seed',
            type=int,
            default=self.config_manager.get_value('training.seed', 33),
            help='Random seed'
        )
        
        train_parser.add_argument(
            '--device',
            type=str,
            default=self.config_manager.get_value('hardware.device', 'auto'),
            help='Device to use (auto, cuda, cpu)'
        )
        
        train_parser.add_argument(
            '--resume',
            type=str,
            default=None,
            help='Path to checkpoint for resuming training'
        )
        
        train_parser.add_argument(
            '--log_dir',
            type=str,
            default=self.config_manager.get_value('logging.log_dir', './logs'),
            help='Directory for saving logs'
        )
        
        train_parser.add_argument(
            '--copy_paste_probability',
            type=float,
            default=self.config_manager.get_value('data.copy_paste_probability', 0.4),
            help='Probability for copy-paste augmentation (0.0 to 1.0)'
        )
        
        train_parser.add_argument(
            '--warmup_epochs',
            type=int,
            default=self.config_manager.get_value('training.warmup_epochs', 5),
            help='Number of warm up epochs for learning rate scheduling'
        )
        
        train_parser.add_argument(
            '--num_classes',
            type=int,
            default=self.config_manager.get_value('model.num_classes', 1),
            help='Number of classes for detection'
        )
    
    def _add_test_parser(self, subparsers: argparse._SubParsersAction):
        """Add testing subcommand parser."""
        test_parser = subparsers.add_parser(
            'test',
            help='Test the trained model',
            description='Evaluate model performance on test dataset'
        )
        
        test_parser.add_argument(
            '--model_path',
            type=str,
            default=self.config_manager.get_value('testing.model_path', './results/best_checkpoint.pth'),
            help='Path to trained model checkpoint'
        )

        test_parser.add_argument(
            '--input_size',
            type=int,
            default=self.config_manager.get_value('testing.input_size', 800),
            help='Input image size'
        )

        test_parser.add_argument(
            '--batch_size',
            type=int,
            default=self.config_manager.get_value('testing.batch_size', 1),
            help='Testing batch size'
        )

        test_parser.add_argument(
            '--device',
            type=str,
            default=self.config_manager.get_value('testing.device', 'auto'),
            help='Device to use for inference'
        )

        test_parser.add_argument(
            '--confidence_threshold',
            type=float,
            default=self.config_manager.get_value('testing.confidence_threshold', 0.5),
            help='Confidence threshold for detections'
        )

        test_parser.add_argument(
            '--num_classes',
            type=int,
            default=self.config_manager.get_value('testing.num_classes', 1),
            help='Number of classes for detection'
        )

        test_parser.add_argument(
            '--backbone',
            type=str,
            default=self.config_manager.get_value('testing.backbone', 'resnet18'),
            help='Backbone architecture (resnet18, resnet50, mobilenet)'
        )

        test_parser.add_argument(
            '--use_tensorrt',
            action='store_true',
            default=self.config_manager.get_value('testing.use_tensorrt', True),
            help='Enable TensorRT optimization for inference'
        )

        test_parser.add_argument(
            '--tensorrt_precision',
            type=str,
            default=self.config_manager.get_value('testing.tensorrt_precision', 'FP16'),
            choices=['FP16', 'FP32', 'INT8'],
            help='TensorRT precision mode (FP16, FP32, INT8)'
        )

        test_parser.add_argument(
            '--log_dir',
            type=str,
            default=self.config_manager.get_value('testing.log_dir', './logs'),
            help='Directory for saving logs'
        )
        
        test_parser.add_argument(
            '--save_visualizations',
            action='store_true',
            default=self.config_manager.get_value('testing.save_visualizations', True),
            help='Save images with bounding boxes and labels'
        )
    
    def _add_deploy_parser(self, subparsers: argparse._SubParsersAction):
        """Add deployment subcommand parser."""
        deploy_parser = subparsers.add_parser(
            'deploy',
            help='Deploy model as API service',
            description='Start FastAPI server with trained model'
        )
        
        deploy_parser.add_argument(
            '--model_path',
            type=str,
            default=self.config_manager.get_value('deployment.model_path', './results/best_checkpoint.pth'),
            help='Path to trained model checkpoint'
        )

        deploy_parser.add_argument(
            '--input_size',
            type=int,
            default=self.config_manager.get_value('model.input_size', 800),
            help='Input image size'
        )

        deploy_parser.add_argument(
            '--host',
            type=str,
            default=self.config_manager.get_value('deployment.host', '0.0.0.0'),
            help='Host to bind API server'
        )

        deploy_parser.add_argument(
            '--port',
            type=int,
            default=self.config_manager.get_value('deployment.port', 8000),
            help='Port to bind API server'
        )

        deploy_parser.add_argument(
            '--device',
            type=str,
            default=self.config_manager.get_value('hardware.device', 'auto'),
            help='Device to use for inference'
        )
    
    def parse_args(self) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args()


class PipelineExecutor:
    """
    Executes the main pipeline based on parsed arguments.
    
    This class handles the execution of different pipeline stages
    (training, testing, deployment) with proper logging and error handling.
    """
    
    def __init__(self, args: argparse.Namespace, arg_parser: ArgumentParser, config_manager: ConfigManager):
        """
        Initialize the pipeline executor.
        
        Args:
            args: Parsed command line arguments
            config_manager: Configuration manager instance
        """
        self.args = args
        self.parser= arg_parser.parser
        self.config_manager = config_manager
        self.logger = None
        self.date_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    
    def setup_logging(self):
        """Initialize logging system."""
        log_dir = getattr(self.args, 'log_dir', self.config_manager.get_value('logging.log_dir', './logs'))
        
        sanity_check(log_dir=log_dir, date_str=self.date_str)
        
        self.logger = Logger(
            log_dir=log_dir,
            log_name="pipeline",
            date_str=self.date_str
        )
    
    def execute(self):
        """
        Execute the appropriate pipeline stage based on command.
        
        Raises:
            ValueError: If invalid command is provided
        """
        if not hasattr(self.args, 'command') or self.args.command is None:
            self.parser.print_help()
            return
        
        self.setup_logging()
        
        try:
            if self.args.command == 'train':
                self._execute_training()
            elif self.args.command == 'test':
                self._execute_testing()
            elif self.args.command == 'deploy':
                self._execute_deployment()
            else:
                raise ValueError(f"Unknown command: {self.args.command}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Pipeline execution failed: {e}")
            else:
                print(f"Error: {e}")
            sys.exit(1)
    
    def _execute_training(self):
        """Execute training pipeline."""
        self.logger.info("Starting training pipeline")
        train_model(args=self.args, logger=self.logger, date_str=self.date_str)
    
    def _execute_testing(self):
        """Execute testing pipeline."""
        self.logger.info("Starting testing pipeline")
        test_model(args=self.args, logger=self.logger, date_str=self.date_str)
    
    def _execute_deployment(self):
        """Execute deployment pipeline."""
        self.logger.info("Starting deployment pipeline")
        run_app(args=self.args, logger=self.logger)


def main():
    """
    Main entry point for the UXO detection pipeline.
    
    This function orchestrates the entire pipeline execution:
    1. Loads configuration from config.yaml
    2. Parses command line arguments
    3. Executes the appropriate pipeline stage
    
    The pipeline supports three main operations:
    - train: Fine-tune the Faster R-CNN model
    - test: Evaluate model performance
    - deploy: Start API server for inference
    """
    
    try:
        config_manager = ConfigManager()

        arg_parser = ArgumentParser(config_manager)
        
        args = arg_parser.parse_args()
        
        executor = PipelineExecutor(args, arg_parser, config_manager)
        executor.execute()
        
    except FileNotFoundError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Configuration parsing error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main()
