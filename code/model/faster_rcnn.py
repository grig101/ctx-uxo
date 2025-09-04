"""
Faster R-CNN model implementation for UXO detection.

This module provides a configurable Faster R-CNN implementation with support
for different ResNet backbones and proper model initialization.
"""

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from typing import Optional, Dict, Any


class FasterRCNNModule(nn.Module):
    """A modular class for building and running Faster R-CNN models with configurable backbones."""

    def __init__(self, logger, num_classes: int, min_size: int = 800, max_size: int = 1280,
                 trainable_backbone_layers: int = 3, pretrained: bool = True,
                 backbone_name: str = "resnet18", rpn_pre_nms_top_n_train: int = 2000,
                 rpn_post_nms_top_n_train: int = 2000, rpn_pre_nms_top_n_test: int = 1000,
                 rpn_post_nms_top_n_test: int = 1000):
        """
        Initializes the Faster R-CNN module.

        Args:
            num_classes (int): Number of object classes (excluding background).
            min_size (int): Minimum size of the input images.
            max_size (int): Maximum size of the input images.
            trainable_backbone_layers (int): Number of trainable layers in the backbone.
            pretrained (bool): Whether to use a backbone pretrained on ImageNet.
            backbone_name (str): Name of the backbone architecture.
            rpn_pre_nms_top_n_train (int): Number of proposals before NMS during training.
            rpn_post_nms_top_n_train (int): Number of proposals after NMS during training.
            rpn_pre_nms_top_n_test (int): Number of proposals before NMS during testing.
            rpn_post_nms_top_n_test (int): Number of proposals after NMS during testing.
        """
        super().__init__()
        self.logger = logger
        self.num_classes = num_classes + 1
        self.min_size = min_size
        self.max_size = max_size
        self.trainable_backbone_layers = trainable_backbone_layers
        self.pretrained = pretrained
        self.backbone_name = backbone_name
        self.rpn_pre_nms_top_n_train = rpn_pre_nms_top_n_train
        self.rpn_post_nms_top_n_train = rpn_post_nms_top_n_train
        self.rpn_pre_nms_top_n_test = rpn_pre_nms_top_n_test
        self.rpn_post_nms_top_n_test = rpn_post_nms_top_n_test
        self.model = self.create()

    def create(self) -> nn.Module:
        """
        Builds the Faster R-CNN model using the specified backbone.

        Returns:
            nn.Module: The constructed Faster R-CNN model.
        """
        valid_backbones = ["resnet18", "resnet50", "mobilenet"]
        if self.backbone_name not in valid_backbones:
            raise ValueError(f"Backbone '{self.backbone_name}' not supported. Use one of {valid_backbones}")

        if self.backbone_name == "mobilenet":
            if self.pretrained:
                self.logger.info("Using MobileNet V3 Large FPN with pretrained weights")
                model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
            else:
                self.logger.info("Using MobileNet V3 Large FPN without pretrained weights")
                model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
            
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            
            return model

        if self.backbone_name == "resnet50":
            if self.pretrained:
                self.logger.info("Using ResNet50 FPN V2 with pretrained weights")
                model = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            else:
                self.logger.info("Using ResNet50 FPN V2 without pretrained weights")
                model = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                
        weights = None
        if self.pretrained:
            if self.backbone_name == "resnet18":
                weights = ResNet18_Weights.DEFAULT
                self.logger.info("Using ResNet18 weights")


        backbone = resnet_fpn_backbone(
            backbone_name=self.backbone_name,
            weights=weights,
            trainable_layers=self.trainable_backbone_layers
        )

        model = FasterRCNN(
            backbone=backbone,
            num_classes=self.num_classes,
            min_size=self.min_size,
            max_size=self.max_size,
            rpn_pre_nms_top_n_train=self.rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=self.rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=self.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=self.rpn_post_nms_top_n_test
        )
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def forward(self, images, targets=None):
        """
        Forward pass of the model.

        Args:
            images (List[Tensor]): A list of input images (C, H, W), normalized.
            targets (List[Dict], optional): Ground truth annotations (used during training).

        Returns:
            During training: Dict[str, Tensor] with loss components.
            During evaluation: List[Dict] with predicted boxes, labels, and scores.
        """
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.
        
        Returns:
            Dict containing model configuration information.
        """
        return {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "trainable_backbone_layers": self.trainable_backbone_layers,
            "pretrained": self.pretrained
        }


def create_model(logger, num_classes: int, backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """
    Factory function to create the Faster R-CNN model.

    Args:
        num_classes (int): Number of classes excluding background.
        backbone (str): Backbone architecture to use (resnet18, resnet50, mobilenet).
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        nn.Module: A model instance ready for training or inference.
    """
    model = FasterRCNNModule(
        logger=logger,
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=pretrained
    )
    
    return model
