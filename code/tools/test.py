"""
Testing module for the UXO detection pipeline.

This module provides comprehensive testing functionality for evaluating
trained models, including ONNX conversion, inference testing, and
performance metrics calculation.
"""

import torch
import onnxruntime as ort
import numpy as np
import time
import warnings
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
from plotnine import ggplot, aes, geom_bar, geom_text, theme_minimal, labs, theme, element_text
from plotnine.themes import theme_bw
from plotnine import element_line, element_rect
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import create_model
from utils.device import DeviceManager
from utils.metrics_plots import MetricsPlotter
from dataset import get_dataset_locations
from src import create_dataloader
from src.data.transforms import get_transforms
from .export_to_onnx import export_onnx, create_tensorrt_session

# Suppress plotnine warnings
warnings.filterwarnings("ignore", category=UserWarning, module="plotnine")
warnings.filterwarnings("ignore", category=FutureWarning, module="plotnine")


class TestConfig:
    """
    Manages testing configuration parameters.
    
    This class centralizes all testing-related configuration
    and provides validation for testing parameters.
    """
    
    def __init__(self, args):
        """
        Initialize testing configuration from arguments.
        
        Args:
            args: Parsed command line arguments
        """
        self.checkpoint = getattr(args, 'checkpoint', getattr(args, 'model_path', './results/best_checkpoint.pth'))
        self.input_size = getattr(args, 'input_size', 800)
        self.device = getattr(args, 'device', 'auto')
        self.batch_size = getattr(args, 'batch_size', 1)
        self.test_batch_size = getattr(args, 'batch_size', 1)  # Use batch_size from args
        self.confidence_threshold = getattr(args, 'confidence_threshold', 0.5)
        self.num_classes = getattr(args, 'num_classes', 1)
        self.backbone = getattr(args, 'backbone', 'resnet50')
        self.use_tensorrt = getattr(args, 'use_tensorrt', True)
        self.tensorrt_precision = getattr(args, 'tensorrt_precision', 'FP16')
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate testing configuration parameters."""
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint}")
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


class TestManager:
    """
    Manages the complete testing process.
    
    This class orchestrates the testing pipeline including model loading,
    ONNX conversion, inference testing, and metrics calculation.
    """
    
    def __init__(self, config: TestConfig, logger, date_str: str):
        """
        Initialize the test manager.
        
        Args:
            config: Testing configuration
            logger: Logger instance
            date_str: Date string for organizing outputs
        """
        self.config = config
        self.logger = logger
        self.date_str = date_str
        self.device_manager = DeviceManager(logger)
        self.device = self.device_manager.get_optimal_device(config.device)
        
        self.model = None
        self.onnx_session = None
        self.test_loader = None
        self.save_visualizations = True
        self.visualization_dir = Path(f"./results/{date_str}/visualizations")
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.app_config = None
        
        self._setup_testing_environment()
    
    def _setup_testing_environment(self):
        """Setup testing environment and optimizations."""
        self.device_manager.setup_device_optimizations(self.device)
        self._load_model()
        self._convert_to_onnx()
        self._create_test_loader()
    
    def _load_model(self):
        """Load the trained model."""
        self.logger.info(f"Loading model from: {self.config.checkpoint}")
        
        self.model = create_model(
            logger=self.logger, 
            num_classes=self.config.num_classes,
            backbone=self.config.backbone
        )
        checkpoint = torch.load(self.config.checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()
        
        self.logger.info(f"Model loaded successfully on {self.device}")
    
    def _convert_to_onnx(self):
        """Convert PyTorch model to ONNX format optimized for TensorRT FP16."""
        self.logger.info("Converting model to ONNX format optimized for TensorRT FP16")
        
        dummy_input = torch.randn((1, 3, self.config.input_size, self.config.input_size)).to(self.device)
        
        onnx_path = export_onnx(
            self.model, 
            dummy_input, 
            f"./results/{self.date_str}/converted.onnx",
            logger=self.logger,
            optimize_for_tensorrt=True
        )
        
        if self.device.type == "cuda" and self.config.use_tensorrt:
            self.logger.info(f"Creating TensorRT session with {self.config.tensorrt_precision} precision")
            self.onnx_session = create_tensorrt_session(
                onnx_path, 
                logger=self.logger, 
                precision=self.config.tensorrt_precision
            )
        else:
            if self.device.type == "cuda":
                self.logger.info("Using CUDA provider (TensorRT disabled)")
                providers = [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_use_max_workspace': '1',
                        'do_copy_in_default_stream': '1',
                    }),
                    'CPUExecutionProvider'
                ]
            else:
                self.logger.info("Using CPU provider (TensorRT not available)")
                providers = ["CPUExecutionProvider"]
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        
        self.logger.info("ONNX conversion and TensorRT optimization completed successfully")
    
    def _create_test_loader(self):
        """Create test data loader."""

        yaml_path = "./dataset/ctxuxo/data.yaml"
        if hasattr(self, 'app_config') and self.app_config:
            yaml_path = self.app_config.get('data', {}).get('dataset', {}).get('yaml_path', yaml_path)
        
        dataset = get_dataset_locations(yaml_path=yaml_path, config=self.app_config)
        transforms = get_transforms(self.config.input_size)
        
        self.test_loader = create_dataloader(
            images_dir=dataset["test"]["images"],
            labels_dir=dataset["test"]["labels"],
            to_shuffle=False,
            batch_size_ds=self.config.test_batch_size,
            input_size=self.config.input_size,
            transform=transforms,
        )
        
        self.logger.info(f"Test loader created with {len(self.test_loader)} batches")
    
    def _preprocess_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Preprocess image for ONNX inference.
        
        Args:
            image: Input image tensor
            
        Returns:
            Preprocessed image as numpy array
        """
        image_np = image.cpu().numpy().astype(np.float32)
        
        if image_np.ndim == 3:
            image_np = np.expand_dims(image_np, axis=0)
        
        return image_np
    
    def _run_inference(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference using ONNX model.
        
        Args:
            image: Input image tensor
            
        Returns:
            Tuple of (boxes, labels, scores)
        """
        input_data = self._preprocess_image(image)
        
        input_name = self.onnx_session.get_inputs()[0].name
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.time()
        outputs = self.onnx_session.run(None, {input_name: input_data})
        inference_time = time.time() - start_time
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        boxes, labels, scores = outputs
        
        return boxes, labels, scores, inference_time
    
    def _filter_predictions(self, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter predictions based on confidence threshold.
        
        Args:
            boxes: Predicted bounding boxes
            labels: Predicted labels
            scores: Prediction scores
            
        Returns:
            Filtered predictions
        """
        valid_indices = scores > self.config.confidence_threshold
        
        filtered_boxes = boxes[valid_indices]
        filtered_labels = labels[valid_indices]
        filtered_scores = scores[valid_indices]
        
        return filtered_boxes, filtered_labels, filtered_scores
    
    def run_testing(self) -> Dict[str, float]:
        """
        Execute the complete testing pipeline.
        
        Returns:
            Dictionary with testing metrics
        """
        self.logger.info("Starting testing pipeline")
        
        total_time = 0.0
        total_images = 0
        batch_times = []
        
        results = []
        categories = [{"id": i, "name": f"class_{i}"} for i in range(1, self.config.num_classes+1)]
        
        coco_dataset = {
            "info": {"description": "UXO Detection Dataset"},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }
        
        image_id = 0
        annotation_id = 0
        
        for batch_idx, (images, targets, image_path) in enumerate(self.test_loader):
            images = list(img.to(self.device) for img in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.model.eval()
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            outputs = self.model(images)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            
            total_time += elapsed
            total_images += len(images)
            
            for i, output in enumerate(outputs):
                current_image_id = image_id + i
                
                image_height, image_width = images[i].shape[1], images[i].shape[2]
                
                coco_dataset["images"].append({
                    "id": current_image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": os.path.basename(image_path[i]) if i < len(image_path) else f"image_{current_image_id}.jpg"
                })
                
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()
                
                # Save visualization with bounding boxes
                self._save_visualization(
                    images[i], boxes, scores, labels, 
                    image_path[i] if i < len(image_path) else f"image_{current_image_id}.jpg",
                    current_image_id
                )
                
                for bbox, score, label in zip(boxes, scores, labels):
                    if label.item() == 0:
                        continue
                        
                    x_min, y_min, x_max, y_max = bbox.tolist()
                    
                    if x_min >= x_max or y_min >= y_max:
                        continue
                        
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    results.append({
                        "image_id": current_image_id,
                        "category_id": int(label.item()),
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score.item())
                    })
                
                labels_gt = targets[i]["labels"].cpu().numpy()
                boxes_gt = targets[i]["boxes"].cpu().numpy()
                
                for label, box in zip(labels_gt, boxes_gt):
                    x_min, y_min, x_max, y_max = box.tolist()
                    
                    if x_min >= x_max or y_min >= y_max:
                        continue
                        
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width <= 0 or height <= 0:
                        continue
                    
                    coco_dataset["annotations"].append({
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": int(label),
                        "bbox": [x_min, y_min, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += len(images)
            batch_times.append(elapsed)
            
            if (batch_idx +1) % 10 == 0 or batch_idx == (len(self.test_loader) - 1):
                avg_time = np.mean(batch_times[-10:])
                self.logger.info(f"[Test] Batch {batch_idx+1}/{len(self.test_loader)} Avg latency: {avg_time:.3f}s")
        
        if len(results) == 0:
            self.logger.warning("No valid predictions found during testing!")
            return {
                'mAP': -1.0, 'AR@100': -1.0, 'avg_time': total_time / max(total_images, 1), 
                'fps': 0.0, 'total_images': total_images, 'total_time': total_time
            }
        
        if len(coco_dataset["annotations"]) == 0:
            self.logger.warning("No ground truth annotations found during testing!")
            return {
                'mAP': -1.0, 'AR@100': -1.0, 'avg_time': total_time / max(total_images, 1), 
                'fps': 0.0, 'total_images': total_images, 'total_time': total_time
            }
        
        try:
            coco = COCO()
            coco.dataset = coco_dataset
            coco.createIndex()
            
            coco_dt = coco.loadRes(results)
            coco_eval = COCOeval(coco, coco_dt, 'bbox')
            coco_eval.params.maxDets = [1, 10, 100]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            mAP = float(coco_eval.stats[0])*100
            mAR = float(coco_eval.stats[8])*100
            
            metrics = {
                'mAP': float(coco_eval.stats[0])*100,  
                'mAP_50': float(coco_eval.stats[1])*100,  
                'mAP_75': float(coco_eval.stats[2])*100,  
                'mAP_small': float(coco_eval.stats[3])*100,  
                'mAP_medium': float(coco_eval.stats[4])*100,  
                'mAP_large': float(coco_eval.stats[5])*100,  
                'mAR_1': float(coco_eval.stats[6])*100, 
                'mAR_10': float(coco_eval.stats[7])*100,  
                'mAR_100': float(coco_eval.stats[8])*100, 
                'mAR_small': float(coco_eval.stats[9])*100,  
                'mAR_medium': float(coco_eval.stats[10])*100, 
                'mAR_large': float(coco_eval.stats[11])*100,  
            }
            
            if self.logger:
                self.logger.info(f"[Test] Predictions: {len(results)}, Ground Truth: {len(coco_dataset['annotations'])}")
                self.logger.info(f"[Test] mAP@0.5:0.95: {mAP:.4f} | AR@100: {mAR:.4f}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during COCO evaluation: {str(e)}")
            mAP = -1.0
            mAR = -1.0
            metrics = {
                'mAP': -1.0, 'mAP_50': -1.0, 'mAP_75': -1.0,
                'mAP_small': -1.0, 'mAP_medium': -1.0, 'mAP_large': -1.0,
                'mAR_1': -1.0, 'mAR_10': -1.0, 'mAR_100': -1.0,
                'mAR_small': -1.0, 'mAR_medium': -1.0, 'mAR_large': -1.0
            }
        
        avg_time = total_time / max(total_images, 1)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        results = {
            'mAP': mAP,
            'AR@100': mAR,
            'avg_time': avg_time,
            'fps': fps,
            'total_images': total_images,
            'total_time': total_time,
            **metrics
        }
        
        self.logger.info(f"[Test] mAP@0.5:0.95: {mAP:.4f} | AR@100: {mAR:.4f}")
        self.logger.info(f"[Test] Latency (avg): {avg_time:.4f}s | FPS: {fps:.2f}")
        
        self._create_testing_plots(results)
        
        return results
    
    def _save_visualization(self, image: torch.Tensor, boxes: torch.Tensor, 
                           scores: torch.Tensor, labels: torch.Tensor, 
                           image_path: str, image_id: int):
        """
        Save image with bounding boxes and labels.
        
        Args:
            image: Input image tensor
            boxes: Predicted bounding boxes
            scores: Confidence scores
            labels: Predicted labels
            image_path: Original image path
            image_id: Image ID for naming
        """
        if not self.save_visualizations:
            return

        img_np = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)

        pil_image = Image.fromarray(img_np)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for box, score, label in zip(boxes, scores, labels):
            if label.item() == 1:  # UXO class
                x1, y1, x2, y2 = box.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                
                # Draw label background
                label_text = f"UXO {score.item():.2f}"
                bbox = draw.textbbox((x1, y1-30), label_text, font=font)
                draw.rectangle(bbox, fill=(255, 0, 0))
                
                # Draw label text
                draw.text((x1, y1-30), label_text, fill=(255, 255, 255), font=font)
        
        # Save the image
        filename = f"result_{image_id:04d}_{Path(image_path).stem}.jpg"
        save_path = self.visualization_dir / filename
        pil_image.save(save_path, quality=95)
        
        self.logger.info(f"Saved visualization: {save_path}")
    
    def _create_testing_plots(self, results: Dict[str, float]):
        """
        Create beautiful ggplot2-style plots for testing results.
        
        Args:
            results: Dictionary with testing results
        """
        plots_dir = f"./results/{self.date_str}/plots_test"
        os.makedirs(plots_dir, exist_ok=True)
        
        plotter = MetricsPlotter(plots_dir, self.logger)
        

        coco_metrics = {
            'mAP': results['mAP'],
            'mAP_50': results.get('mAP_50', -1.0),
            'mAP_75': results.get('mAP_75', -1.0),
            'mAR_1': results.get('mAR_1', -1.0),
            'mAR_10': results.get('mAR_10', -1.0),
            'mAR_100': results['AR@100'],
        }
        
        plotter.plot_coco_metrics_breakdown(coco_metrics)
        plotter.plot_performance_comparison(results)
        plotter.plot_object_size_metrics_test(results)
        plotter.plot_comprehensive_test_metrics(results)

        results_df = pd.DataFrame([results])
        csv_path = f"{plots_dir}/test_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Testing plots saved to {plots_dir}")


def test_model(args, logger, date_str: str) -> Dict[str, float]:
    """
    Main testing function.
    
    This function serves as the entry point for the testing pipeline,
    creating the necessary components and executing the testing process.
    
    Args:
        args: Parsed command line arguments
        logger: Logger instance
        date_str: Date string for organizing outputs
        
    Returns:
        Dictionary with testing results
    """
    config = TestConfig(args)
    manager = TestManager(config, logger, date_str)
    

    try:
        import yaml
        with open('config.yaml', 'r') as f:
            app_config = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load config.yaml: {e}")
        app_config = None
    

    manager.app_config = app_config
    
 
    manager._create_test_loader()
    
    return manager.run_testing()
