""" We created a custom dataset loader which convert CTX-UXO DS 
 to pytorch compatible bbox (4 coords) format
 In this work, all classes are relabelled for a binary classication UXO/NON-UXO

 Note!!! You should ALWAYS add the background class (in this case,
 the final num_classes used in our model instances will be num_classes+1)
 Background class is always index 0
"""
import os
import torch
import cv2
from .utils import polygon_to_bbox, clamp
from PIL import Image
import random
num_classes=1 # UXO/NON-UXO . 1 class + 1

"""
YOLO POLYGONS EXAMPLE

"""

def collate_fn(batch):
    return tuple(zip(*batch))

def extract_polygon(polygon, image_shape, x_scale=None, y_scale=None):
    coords = []
    for i in range(0, len(polygon), 2):
        x_pixel = polygon[i] * image_shape[0]
        y_pixel = polygon[i + 1] * image_shape[1]
        if x_scale and y_scale:
            x_pixel = x_pixel * x_scale
            y_pixel = y_pixel * y_scale
        # coords.append((x_pixel, y_pixel)) # if you want as pairs
        coords.append(x_pixel)
        coords.append(y_pixel)
        
    return coords

# Convert YOLO for TorchVision Custom Dataset (see class CTXUXODataset)
# More exactly, this is a scaling method for bboxes.
def load_yolo_anno(label_path, image_shape, input_size):
   
    """
    Args:
        label_path - path to the .txt file with annotations
        image_shape - original shape of the image (before transforms)
    Returns:
        boxes, labels - filtered bounding boxes and their corresponding labels
    """
    labels = []
    boxes = []

    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            class_id = int(data[0])

            polygon = list(map(float, data[1:]))
            x_scale = input_size / image_shape[0]
            y_scale = input_size / image_shape[1]
            coords =extract_polygon(polygon, image_shape, x_scale, y_scale)   
            bbox = polygon_to_bbox(coords)
  
            bbox = [
                clamp(bbox[0], 0, input_size),
                clamp(bbox[1], 0, input_size),
                clamp(bbox[2], 0, input_size),
                clamp(bbox[3], 0, input_size),
            ]

            boxes.append(bbox)
            labels.append(class_id)

    return boxes, labels

# Custom Dataset for COCO/YOLO
class CTXUXODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, input_size, copy_paste,  transform=None):
        """
    Custom Dataset Loader
    See: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args: images_dir - location of images
          labels_dir - locations of labels (.txt file with annotations
          )

    Returns:
          by __getitem__ method
          image - tensor image with transforms applied
          targets - is a dict() with 2 keys: boxes and labels

          The batch image will be a list of image tensors.
          The batch targets will be a list of targets dicts.
          Check collate_fn() function.
    """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.copy_paste = copy_paste
        self.input_size = input_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        max_attempts = len(self.image_files)
        attempts = 0
        original_idx = idx
        
        while attempts < max_attempts:
            image_path = os.path.join(self.images_dir, self.image_files[idx])
            label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')

            image = cv2.imread(image_path)
            if image is None:
                import warnings
                warnings.warn(f"[WARN] Failed to load image: {image_path}. Skipping.")
                idx = (idx + 1) % len(self.image_files)
                attempts += 1
                
                if idx == original_idx:
                    raise RuntimeError(f"Failed to load any images after {max_attempts} attempts")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            break 
        
        original_shape = image.shape[:2]

        boxes, labels = load_yolo_anno(label_path, original_shape, self.input_size)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        labels=[1 for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        epsilon = 1e-6
        if self.copy_paste:
            if self.copy_paste.p > epsilon:
                all_coords= []
                x_scale = self.input_size / original_shape[0]
                y_scale = self.input_size / original_shape[1]
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        data = line.strip().split()
                        polygon = list(map(float, data[1:]))
                        coords = extract_polygon(polygon, original_shape, x_scale, y_scale)                    
                        all_coords.append(coords)
                    
                image_np = image.copy()  
                labels_list = labels

                status, aug_image, aug_boxes, aug_labels = self.copy_paste(image_np, all_coords, labels_list, self.input_size)
            
                if status==True:
                    image = aug_image
                    boxes = torch.tensor(aug_boxes, dtype=torch.float32)
                    labels = torch.tensor(aug_labels, dtype=torch.int64)

        # debug_image = image.copy()
        # debug_image=cv2.resize(debug_image,(self.input_size, self.input_size))
        # for bbox, label in zip(boxes, labels):
        #     x1, y1, x2, y2 = map(int, bbox.tolist())
        #     cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(debug_image, str(label.item()), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # debug_path = f"./dataset/debug/{os.path.splitext(os.path.basename(image_path))[0]}_dbg_np_{random.randint(1,9999)}.jpg"
        # os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        # Image.fromarray(debug_image).save(debug_path)
        if self.transform:
            image=self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels
        }
        return image, target, image_path

def test_ds(ds, count_samples, logger=None):
    """
    Test dataset functionality.
    
    Args:
        ds: Dataset to test
        count_samples: Number of samples to test
        logger: Logger instance for output
    """
    for count, (image, target, image_path) in enumerate(ds):
        if count >= count_samples:
            break
        if logger:
            logger.info(f"Image path: {image_path}, shape: {image.shape}, Target: {target}")
        else:
            print(f"Image path: {image_path}, shape: {image.shape}, Target: {target}")


def create_dataloader(images_dir, labels_dir, to_shuffle, batch_size_ds,
                      input_size, copy_paste= None,  transform=None):
    """
    Create a dataloder
    images_dir: image location (relative or absolute path)
    labels_dir
    """
#   image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    subset= CTXUXODataset(images_dir, labels_dir, input_size, copy_paste, transform=transform) # train/val/test
    dataloader = torch.utils.data.DataLoader(subset,
                                            batch_size=batch_size_ds,
                                            shuffle=to_shuffle,
                                            collate_fn=collate_fn)
    return dataloader


""" 
The following commented code is for splitting the dataset into train, val, test.
You can use it if you want to split the dataset into train, val, test, but is
highly recommended to use the dataset as it is.

"""

# def create_datasets(images_dir, labels_dir, transform, train_split=0.7, val_split=0.9):
#     """ Splitting the dataset in 3 parts (train, val, test)
#     images_dir - location of images
#     labels_dir - location of labels (*.txt)
#     transform - transform method from torchvision
#     train_split - split percetange (0.x train dataset)
#     val_split - split percentage (val_split-train_split)

#     Note: test_split = 1 - train_split - val_split

#     """
#     image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

#     num_train = int(len(image_files) * train_split)
#     num_val = int(len(image_files) * val_split)

#     train_dataset = CTXUXODataset(images_dir, labels_dir, transform=transform)
#     val_dataset = CTXUXODataset(images_dir, labels_dir, transform=transform)
#     test_dataset= CTXUXODataset(images_dir, labels_dir, transform=transform)

#     train_dataset.image_files = image_files[:num_train]
#     val_dataset.image_files = image_files[num_train:num_val]
#     test_dataset.image_files = image_files[num_val:]

#     return train_dataset, val_dataset, test_dataset


# images_dir, labels_dir = get_dataset()

# train_dataset, val_dataset, test_dataset = create_datasets(images_dir,
#                                                             labels_dir,
#                                                             transform,
#                                                             train_split=0.7,
#                                                             val_split=0.85)
#                                                                         # test will be
#                                                                         # 1 - train_split
#                                                                         #   - valid_split
#                                                             # The label which is not binary_label
#                                                             # will be replace by "UXO" [1]




# train_data_loader = torch.utils.data.DataLoader(train_dataset,
#                                                 batch_size=batch_size_ds,
#                                                 shuffle=True,
#                                                 collate_fn=collate_fn )


# val_data_loader = torch.utils.data.DataLoader(val_dataset,
#                                             batch_size=batch_size_ds,
#                                             shuffle=False,
#                                             collate_fn=collate_fn )

# test_data_loader = torch.utils.data.DataLoader(test_dataset,
#                                             batch_size=batch_size_ds,
#                                             shuffle=False,
#                                             collate_fn=collate_fn )

