"""
Training module for the UXO detection pipeline.

This module provides  training functionality for the Faster R-CNN
model, including data loading, model training, validation, and checkpointing.
"""

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import os
import sys
import random
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from plotnine import (
    ggplot, aes,
    geom_line, geom_point,
    facet_wrap,
    labs,
    theme, theme_minimal,
    element_text, element_line, element_rect,
    scale_color_manual
)
from plotnine.themes import theme_bw
from model import create_model
from utils.device import DeviceManager
from utils.metrics_plots import create_comprehensive_plots
from src import create_dataloader
from src.data.transforms import get_transforms, CustomCopyPasteWithBackground
from .val import validation


warnings.filterwarnings("ignore", category=UserWarning, module="plotnine")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotnine")


class TrainingConfig:
    """Manages training configuration parameters."""

    def __init__(self, args):
        """Initialize training configuration from arguments."""
        self.num_epochs = getattr(args, 'num_epochs', 300)
        self.learning_rate = getattr(args, 'learning_rate', 0.005)
        self.batch_size = getattr(args, 'batch_size', 16)
        self.test_batch_size = getattr(args, 'test_batch_size', 1)
        self.weight_decay = getattr(args, 'weight_decay', 0.0001)
        self.momentum = getattr(args, 'momentum', 0.9)
        self.seed = getattr(args, 'seed', 33)
        self.early_stopping_epochs = getattr(args, 'early_stopping_epochs', 50)
        self.input_size = getattr(args, 'input_size', 800)
        self.backbone = getattr(args, 'backbone', 'resnet50')
        self.device = getattr(args, 'device', 'auto')
        self.resume = getattr(args, 'resume', None)
        self.copy_paste_probability = getattr(args, 'copy_paste_probability', 0.4)
        self.warmup_epochs = getattr(args, 'warmup_epochs', 5)
        self.num_classes = getattr(args, 'num_classes', 1)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate training configuration."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.early_stopping_epochs < 0:
            raise ValueError("early_stopping_epochs must be non-negative")
        if not 0 <= self.copy_paste_probability <= 1:
            raise ValueError("copy_paste_probability must be between 0 and 1")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.warmup_epochs >= self.num_epochs:
            raise ValueError("warmup_epochs must be less than num_epochs")


class TrainingManager:
    """Manages the complete training process."""
    
    def __init__(self, config: TrainingConfig, logger, date_str: str):
        """
        Initialize the training manager.

    Args:
            config: Training configuration
            logger: Logger instance
            date_str: Date string for organizing outputs
        """
        self.config = config
        self.logger = logger
        self.date_str = date_str
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.writer = None
        self.app_config = None
        self.plateau_scheduler = None
        self.num_classes = config.num_classes
        
        self._setup_training_environment()
        self._set_random_seed()
        self._setup_logging()
        self._create_model()
        self._create_optimizer()
        self._create_data_loaders()
        
        if self.config.resume:
            self._load_checkpoint(self.config.resume)
    
    def _setup_training_environment(self):
        """Setup training environment and device."""
        device_manager = DeviceManager(self.logger)
        self.device = device_manager.get_optimal_device(self.config.device)
        device_manager.setup_device_optimizations(self.device)
        self.logger.info(f"Using device: {self.device}")
    
    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _setup_logging(self):
        """Setup tensorboard logging."""
        log_dir = f"./logs/{self.date_str}/tensorboard"
        self.writer = SummaryWriter(log_dir)
    
    def _create_model(self):
        """Create and setup the model."""
        self.model = create_model(
            logger=self.logger,
            num_classes=self.config.num_classes,
            backbone=self.config.backbone,
            pretrained=True
        )
        self.model.to(self.device)
        
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)
    
    def _create_optimizer(self):
        """Create optimizer and scheduler with warm up."""
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.logger.info(f"Decay params: {len(decay_params)}")
        self.logger.info(f"No decay params: {len(no_decay_params)}")
        
        self.optimizer = optim.SGD(
            [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
            ],
            lr=self.config.learning_rate,
            momentum=self.config.momentum
        )
        
        def warmup_lambda(epoch):
            if epoch < self.config.warmup_epochs:
                return (epoch + 1) / self.config.warmup_epochs
            return 1.0
        
        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
            )
        self.logger.info(f"Created warm up scheduler for {self.config.warmup_epochs} epochs")
        self.logger.info("Created ReduceLROnPlateau scheduler for validation loss")
    
    def _load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load training checkpoint.

    Args:
            checkpoint_path: Path to checkpoint file

    Returns:
            True if checkpoint loaded successfully
        """
        if not os.path.exists(checkpoint_path):
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'warmup_scheduler_state_dict' in checkpoint:
                self.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
            if 'plateau_scheduler_state_dict' in checkpoint:
                self.plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler_state_dict'])
            
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            self.logger.info(f"Loaded checkpoint from epoch {self.start_epoch}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def _create_data_loaders(self):
        """Create training and validation data loaders."""
        from dataset import get_dataset_locations
        
        yaml_path = "./dataset/ctxuxo/data.yaml"
        if hasattr(self, 'app_config') and self.app_config:
            yaml_path = self.app_config.get('data', {}).get('dataset', {}).get('yaml_path', yaml_path)
        
        dataset = get_dataset_locations(yaml_path=yaml_path, config=self.app_config)
        transforms = get_transforms(self.config.input_size)
        
        copy_paste_aug = CustomCopyPasteWithBackground(
            p=self.config.copy_paste_probability,
            logger=self.logger
        )
        
        self.train_loader = create_dataloader(
        images_dir=dataset["train"]["images"],
        labels_dir=dataset["train"]["labels"],
        to_shuffle=True,
        batch_size_ds=self.config.batch_size,
        input_size=self.config.input_size,
        copy_paste=copy_paste_aug,
        transform=transforms,
        )
        
        self.val_loader = create_dataloader(
        images_dir=dataset["val"]["images"],
        labels_dir=dataset["val"]["labels"],
        to_shuffle=False,
        batch_size_ds=self.config.batch_size,
        input_size=self.config.input_size,
        copy_paste=None,
        transform=transforms,
        )
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, targets, image_paths) in enumerate(self.train_loader):
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx +1) % 10 == 0 or batch_idx == (num_batches - 1):
                self.logger.info(f'Epoch {epoch}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.logger.info(f'Epoch {epoch} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        return {'train_loss': avg_loss, 'learning_rate': current_lr}
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        avg_loss, mAP, mAR, avg_time, fps, metrics = validation(self.model, self.val_loader, self.device, self.num_classes, logger=self.logger)
        
        return {
            'val_loss': avg_loss,
            'mAP': mAP,
            'mAR': mAR,
            'avg_time': avg_time,
            'fps': fps,
            **metrics
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'plateau_scheduler_state_dict': self.plateau_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_dir = f"./results/{self.date_str}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_checkpoint_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def _save_training_history(self, history: list):
        """Save training history to CSV."""
        checkpoint_dir = f"./results/{self.date_str}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        csv_path = os.path.join(checkpoint_dir, 'loss_history.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'val_loss', 'learning_rate', 
                         'mAP', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large',
                         'mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large',
                         'avg_time', 'fps']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in history:
                writer.writerow(entry)
    
    def _create_comprehensive_plots(self, history: list, final_metrics: dict):
        """Create plots including COCO metrics."""
        checkpoint_dir = f"./results/{self.date_str}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        plots_dir = os.path.join(checkpoint_dir, 'plots_val')
        os.makedirs(plots_dir, exist_ok=True)
        
        create_comprehensive_plots(history, final_metrics, plots_dir, self.logger)
    
    def _run_test_evaluation(self):
        """Run test evaluation using the best checkpoint."""
        from .test import test_model
        
        checkpoint_dir = f"./results/{self.date_str}"
        best_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        
        if not os.path.exists(best_checkpoint_path):
            self.logger.warning(f"Last checkpoint not found at {best_checkpoint_path}")
            return
        
        self.logger.info("Running test evaluation with best checkpoint...")
        
        class TestArgs:
            def __init__(self, config):
                self.checkpoint = best_checkpoint_path
                self.input_size = config.input_size
                self.batch_size = config.test_batch_size
                self.device = config.device
                self.num_classes = config.num_classes
                self.backbone = config.backbone
                self.confidence_threshold = 0.05
        
        test_args = TestArgs(self.config)
        
        # Run test evaluation
        test_model(test_args, self.logger, self.date_str)
        
        self.logger.info("Test evaluation completed!")
    
    def _create_training_plots(self, history: list):
        """Create training plots using plotnine."""
        if not history:
            return
        self.logger.info(f"Creating training plots for {len(history)} epochs")
        df = pd.DataFrame(history)
        
        checkpoint_dir = f"./results/{self.date_str}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        df_melted = df.melt(id_vars=['epoch'], value_vars=['train_loss', 'val_loss'], 
                           var_name='loss_type', value_name='loss_value')
        df_melted['loss_type'] = df_melted['loss_type'].map({'train_loss': 'Training Loss', 'val_loss': 'Validation Loss'})
        
        loss_plot = (
            ggplot(df_melted, aes(x='epoch', y='loss_value', color='loss_type')) +
            geom_line(size=1) +
            geom_point(size=2) +
            scale_color_manual(values=["#FF4444", "#4444FF"]) +
            labs(
                title="Training and Validation Loss",
                x="Epoch",
                y="Loss",
                color="Loss Type"
            ) +
            theme_minimal() +
            theme(
                plot_title=element_text(size=16, weight="bold"),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                legend_title=element_text(size=12, weight="bold"),
                legend_text=element_text(size=11),
                panel_grid_major=element_line(color="#E5E5E5", alpha=0.3),
                panel_grid_minor=element_line(color="#F2F2F2", alpha=0.1)
            )
        )
        
        lr_plot = (
            ggplot(df, aes(x='epoch', y='learning_rate')) +
            geom_line(color="#4444FF", size=1) +
            geom_point(color="#4444FF", size=2) +
            labs(
                title="Learning Rate Schedule",
                x="Epoch",
                y="Learning Rate"
            ) +
            theme_minimal() +
            theme(
                plot_title=element_text(size=16, weight="bold"),
                axis_title=element_text(size=13, weight="bold"),
                axis_text=element_text(size=11),
                panel_grid_major=element_line(color="#E5E5E5", alpha=0.3),
                panel_grid_minor=element_line(color="#F2F2F2", alpha=0.1)
            )
        )
        
        loss_plot.save(os.path.join(checkpoint_dir, 'loss_plot.png'), dpi=300, width=10, height=6)
        lr_plot.save(os.path.join(checkpoint_dir, 'lr_plot.png'), dpi=300, width=10, height=6)
        
        self.logger.info("Training plots saved")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.num_epochs} epochs")
        self.logger.info(f"Warm up for {self.config.warmup_epochs} epochs")
        
        history = []
        patience_counter = 0
        final_metrics = {}
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            metrics = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics['val_loss'],
                'learning_rate': current_lr,
                'mAP': val_metrics.get('mAP', -1.0),
                'mAP_50': val_metrics.get('mAP_50', -1.0),
                'mAP_75': val_metrics.get('mAP_75', -1.0),
                'mAP_small': val_metrics.get('mAP_small', -1.0),
                'mAP_medium': val_metrics.get('mAP_medium', -1.0),
                'mAP_large': val_metrics.get('mAP_large', -1.0),
                'mAR_1': val_metrics.get('mAR_1', -1.0),
                'mAR_10': val_metrics.get('mAR_10', -1.0),
                'mAR_100': val_metrics.get('mAR_100', -1.0),
                'mAR_small': val_metrics.get('mAR_small', -1.0),
                'mAR_medium': val_metrics.get('mAR_medium', -1.0),
                'mAR_large': val_metrics.get('mAR_large', -1.0),
                'avg_time': val_metrics.get('avg_time', 0.0),
                'fps': val_metrics.get('fps', 0.0)
            }
            
            history.append(metrics)
            final_metrics = val_metrics
            
            self.writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            self.writer.add_scalar('Metrics/mAP', val_metrics.get('mAP', -1.0), epoch)
            self.writer.add_scalar('Metrics/mAP_50', val_metrics.get('mAP_50', -1.0), epoch)
            self.writer.add_scalar('Metrics/mAP_75', val_metrics.get('mAP_75', -1.0), epoch)
            self.writer.add_scalar('Metrics/mAR_100', val_metrics.get('mAR_100', -1.0), epoch)
            
            is_best = val_metrics['val_loss'] < self.best_val_loss
            
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self._save_checkpoint(epoch, is_best)
            
            if epoch < self.config.warmup_epochs:
                self.warmup_scheduler.step()
                self.logger.info(f"Warm up epoch {epoch + 1}/{self.config.warmup_epochs}, LR: {current_lr:.6f}")
            else:
                self.plateau_scheduler.step(val_metrics['val_loss'])
                self.logger.info(f"Plateau scheduler step with val_loss: {val_metrics['val_loss']:.4f}, LR: {current_lr:.6f}")
            
            if patience_counter >= self.config.early_stopping_epochs:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
            self._save_training_history(history)
            self._create_training_plots(history)
            self._create_comprehensive_plots(history, final_metrics)

        self.writer.close()
        
        self._run_test_evaluation()
        
        self.logger.info("Training completed!")


def train_model(args, logger, date_str: str):
    """
    Main training function.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        date_str: Date string for organizing outputs
    """
    config = TrainingConfig(args)
    trainer = TrainingManager(config, logger, date_str)
    trainer.train()


