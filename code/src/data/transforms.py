"""
Data augmentation transforms for UXO detection.

This module provides custom transforms including copy-paste augmentation
with random scaling and rotation of segmented instances.
"""

import random
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import torchvision.transforms as T
import requests
from .utils import polygon_to_bbox, clamp
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CustomCopyPasteWithBackground:
    """
    Custom copy-paste augmentation with random scaling and rotation.
    
    This transform takes segmented instances (polygons) and applies:
    - Random scaling between scale_min and scale_max
    - Random rotation between angle_min and angle_max
    - Copy-paste onto random backgrounds
    """
    
    def __init__(self, backgrounds_dir="./dataset/backgrounds", p=0.4, 
                 scale_min=0.5, scale_max=1.2, angle_min=-0.3, angle_max=0.3, 
                 logger=None):
        """
        Initialize the custom copy-paste augmentation.
        
        Args:
            backgrounds_dir: Directory containing background images
            p: Probability of applying the augmentation
            scale_min: Minimum scaling factor
            scale_max: Maximum scaling factor
            angle_min: Minimum rotation angle in radians
            angle_max: Maximum rotation angle in radians
            logger: Logger instance for output
        """
        
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.angle_min = angle_min
        self.angle_max = angle_max
        
        if logger:
            if p > 0:
                logger.info(f"Init copy paste augmentation. Probability: {p}")
                logger.info(f"Scale range: [{scale_min}, {scale_max}], Angle range: [{angle_min}, {angle_max}]")
            else:
                logger.info("Copy paste augmentation is disabled")
                return 
            
        if not os.path.exists(backgrounds_dir) or (os.path.exists(backgrounds_dir) and not os.listdir(backgrounds_dir)):
            if logger:
                logger.info("Downloading dataset ADE20K for backgrounds")
            try:
                os.makedirs(backgrounds_dir, exist_ok=True)
                

                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                backgrounds = load_dataset("1aurent/ADE20K", split="train[:10%]")
                for i, sample in enumerate(tqdm(backgrounds, desc="Saving images")):
                    image_url = sample["image"]

                    if isinstance(image_url, str):
                        response = session.get(image_url, stream=True, timeout=30)
                        response.raise_for_status()
                        image = Image.open(response.raw).convert("RGB")
                    else:
                        image = image_url.convert("RGB")

                    image.save(os.path.join(backgrounds_dir, f"ade20k_bg_{i:05d}.jpg"))
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to download ADE20K backgrounds: {e}")
                    logger.info("Continuing without background augmentation")
                self.backgrounds = []
                return

        self.backgrounds = [os.path.join(backgrounds_dir, f)
                            for f in os.listdir(backgrounds_dir)
                            if f.endswith('.jpg') or f.endswith('.png')]

    def _transform_polygon(self, polygon, scale_factor, angle, center_x, center_y):
        """
        Apply scaling and rotation to a polygon.
        
        Args:
            polygon: List of [x, y] points
            scale_factor: Scaling factor
            angle: Rotation angle in radians
            center_x: Center x coordinate for transformation
            center_y: Center y coordinate for transformation
            
        Returns:
            Transformed polygon points
        """
        if len(polygon) < 3:
            return polygon
            
        polygon = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        polygon[:, 0] -= center_x
        polygon[:, 1] -= center_y

        polygon *= scale_factor

        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        polygon[:, 0] = x_coords * cos_a - y_coords * sin_a
        polygon[:, 1] = x_coords * sin_a + y_coords * cos_a

        polygon[:, 0] += center_x
        polygon[:, 1] += center_y
        
        return polygon.astype(np.int32).reshape(-1).tolist()

    def _get_polygon_center(self, polygon):
        """
        Calculate the center of a polygon.
        
        Args:
            polygon: List of [x, y] points
            
        Returns:
            Tuple of (center_x, center_y)
        """
        if len(polygon) < 2:
            return 0, 0
            
        polygon = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        center_x = np.mean(polygon[:, 0])
        center_y = np.mean(polygon[:, 1])
        
        return center_x, center_y

    def __call__(self, image, polygons, labels, input_size):
        """
        Apply copy-paste augmentation with scaling and rotation.
        
        Args:
            image: Input image
            polygons: List of polygon points for each instance
            labels: List of labels for each instance
            input_size: Size of the input image
            
        Returns:
            Tuple of (status, transformed_image, new_bbox, new_labels)
        """
        status = False
        if random.random() > self.p or len(polygons) == 0:
            return status, image, polygons, labels

        h_img = input_size
        w_img = input_size

        bg_path = random.choice(self.backgrounds)
        bg = cv2.imread(bg_path)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        bg_resized = cv2.resize(bg, (w_img, h_img))
        image = cv2.resize(image, (w_img, h_img))
        pasted_image = bg_resized.copy()
        
        new_bbox = []
        new_labels = []
       
        for poly, label in zip(polygons, labels):
            if len(poly) < 6:
                continue
                
            scale_factor = random.uniform(self.scale_min, self.scale_max)
            angle = random.uniform(self.angle_min, self.angle_max)

            center_x, center_y = self._get_polygon_center(poly)
            
            pts_orig = np.array(poly, dtype=np.int32).reshape(-1, 2)
            mask_orig = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(mask_orig, [pts_orig], color=1)
            
            obj_orig = cv2.bitwise_and(image, image, mask=mask_orig)
            
            bbox_orig = polygon_to_bbox(poly)
            x1, y1, x2, y2 = bbox_orig
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            padding = 20
            x1_pad = max(0, x1 - padding)
            y1_pad = max(0, y1 - padding)
            x2_pad = min(w_img, x2 + padding)
            y2_pad = min(h_img, y2 + padding)
            
            y1_pad, y2_pad, x1_pad, x2_pad = int(y1_pad), int(y2_pad), int(x1_pad), int(x2_pad)
            
            obj_region = obj_orig[y1_pad:y2_pad, x1_pad:x2_pad]
            mask_region = mask_orig[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if obj_region.size == 0:
                continue
                
            region_center_x = (x2_pad - x1_pad) // 2
            region_center_y = (y2_pad - y1_pad) // 2
            
            rotation_matrix = cv2.getRotationMatrix2D(
                (region_center_x, region_center_y), 
                np.degrees(angle), 
                scale_factor
            )
            
            obj_rotated = cv2.warpAffine(obj_region, rotation_matrix, 
                                        (obj_region.shape[1], obj_region.shape[0]))
            mask_rotated = cv2.warpAffine(mask_region, rotation_matrix, 
                                         (mask_region.shape[1], mask_region.shape[0]))
            
            transformed_poly = self._transform_polygon(poly, scale_factor, angle, center_x, center_y)
            
            if len(transformed_poly) < 6:
                continue
                
            pts = np.array(transformed_poly, dtype=np.int32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)
            
            mask_transformed = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(mask_transformed, [pts], color=1)
            
            if mask_transformed.sum() == 0:
                continue
                
            bbox_transformed = polygon_to_bbox(transformed_poly)
            x1_new, y1_new, x2_new, y2_new = bbox_transformed
            
            x1_new, y1_new, x2_new, y2_new = int(x1_new), int(y1_new), int(x2_new), int(y2_new)
            
            if x2_new <= x1_new or y2_new <= y1_new:
                continue
                
            try:
                target_width = int(x2_new - x1_new)
                target_height = int(y2_new - y1_new)
                
                if target_width <= 0 or target_height <= 0:
                    continue
                    
                obj_resized = cv2.resize(obj_rotated, (target_width, target_height))
                mask_resized = cv2.resize(mask_rotated, (target_width, target_height))
                
                obj_height = int(obj_resized.shape[0])
                obj_width = int(obj_resized.shape[1])
                
                if y1_new + obj_height <= h_img and x1_new + obj_width <= w_img:
                    region_to_paste = pasted_image[y1_new:y1_new + obj_height, 
                                                 x1_new:x1_new + obj_width]
                    
                    if len(mask_resized.shape) == 2:
                        mask_region = mask_resized[:, :, None]
                    else:
                        mask_region = mask_resized
                    
                    if mask_region.shape[:2] == obj_resized.shape[:2] and obj_resized.shape[:2] == region_to_paste.shape[:2]:
                        pasted_image[y1_new:y1_new + obj_height, 
                                   x1_new:x1_new + obj_width] = \
                            np.where(mask_region == 1, obj_resized, region_to_paste)
                    else:
                        continue
                    
                    bbox = [
                        clamp(x1_new, 0, input_size),
                        clamp(y1_new, 0, input_size),
                        clamp(x2_new, 0, input_size),
                        clamp(y2_new, 0, input_size),
                    ]
                    
                    new_bbox.append(bbox)
                    new_labels.append(label)
                    status = True
                    
            except Exception as e:
                print(f"Error processing object: {e}")
                continue

        return status, pasted_image, new_bbox, new_labels


def get_transforms(input_size):
    """
    Get standard transforms for image preprocessing.
    
    Args:
        input_size: Target image size
        
    Returns:
        Compose transform
    """
    resulted_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return resulted_transforms