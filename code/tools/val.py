"""
Validation module for UXO detection model.

This module provides validation functionality including
COCO evaluation metrics calculation and proper error handling.
"""

import torch
import time
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Dict, List, Tuple, Optional


@torch.no_grad()
def validation(model, val_loader, device, num_classes, writer=None, epoch=None, logger=None):
    """
    Perform validation on the model using COCO evaluation metrics.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device to run validation on
        num_classes: Number of classes (excluding background)
        writer: TensorBoard writer for logging
        epoch: Current epoch number
        logger: Logger instance for output

    Returns:
        Tuple containing (avg_loss, mAP, mAR, avg_time, fps)
    """
    logger.info(f"[Validation] Starting validation...")
    total_loss = 0
    num_batches = 0
    total_time = 0.0
    total_images = 0

    results = []
    categories = [{"id": i, "name": f"class_{i}"} for i in range(1, num_classes+1)]
    

    coco_dataset = {
        "info": {"description": "UXO Detection Dataset"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 0
    annotation_id = 0

    for batch_idx, (images, targets, image_path) in enumerate(val_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())
        total_loss += loss.item()
        num_batches += 1
        model.eval()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        outputs = model(images)
        if device == "cuda":
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
        
        if writer is not None and epoch is not None:
            writer.add_scalar('Val/Batch_Loss', loss.item(), epoch * len(val_loader) + batch_idx)

        if (batch_idx +1)  % 10 == 0 or batch_idx == (len(val_loader) - 1):
            avg_batch_time = total_time / total_images if total_images > 0 else 0
            fps = 1.0 / avg_batch_time if avg_batch_time > 0 else 0.0
            logger.info(f"[Validation] Batch {batch_idx+1}/{len(val_loader)} Loss: {loss.item():.4f} | "
                        f"Batch Latency: {elapsed:.3f}s | Avg Latency: {avg_batch_time:.4f}s | FPS: {fps:.2f}")
        model.train()

        
    if len(results) == 0:
        if logger:
            logger.warning("No valid predictions found during validation!")
        return total_loss / max(num_batches, 1), -1.0, -1.0, total_time / max(total_images, 1), 0.0, {}

    if len(coco_dataset["annotations"]) == 0:
        if logger:
            logger.warning("No ground truth annotations found during validation!")
        return total_loss / max(num_batches, 1), -1.0, -1.0, total_time / max(total_images, 1), 0.0, {}

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

        if logger:
            logger.info(f"[Validation] Predictions: {len(results)}, Ground Truth: {len(coco_dataset['annotations'])}")
            logger.info(f"[Validation] mAP@0.5:0.95: {mAP:.4f} | AR@100: {mAR:.4f}")

    except Exception as e:
        if logger:
            logger.error(f"Error during COCO evaluation: {str(e)}")
        mAP = -1.0
        mAR = -1.0
        metrics = {
            'mAP': -1.0, 'mAP_50': -1.0, 'mAP_75': -1.0,
            'mAP_small': -1.0, 'mAP_medium': -1.0, 'mAP_large': -1.0,
            'mAR_1': -1.0, 'mAR_10': -1.0, 'mAR_100': -1.0,
            'mAR_small': -1.0, 'mAR_medium': -1.0, 'mAR_large': -1.0
        }

    avg_loss = total_loss / max(num_batches, 1)
    avg_time = total_time / max(total_images, 1)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    if logger:
        logger.info(f"[Validation] Avg Loss: {avg_loss:.4f} | mAP@0.5:0.95: {mAP:.4f} | AR@100: {mAR:.4f}")
        logger.info(f"[Validation] Latency (avg): {avg_time:.4f}s | FPS: {fps:.2f}")
    
    model.train()

    return avg_loss, mAP, mAR, avg_time, fps, metrics
