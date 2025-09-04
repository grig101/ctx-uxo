import cv2
import matplotlib.pyplot as plt

def show_samples(dataset_loader, images_to_show=5, input_size=800):
    """
    Args:
        dataset_loader (DataLoader): PyTorch DataLoader for the dataset.
        images_to_show (int): Number of images to display.
        input_size (tuple): Tuple specifying the width and height for resizing images.

    Returns:
        None
    """
    dataset_classes = {
        0: "background",
        1: "UXO"
    }

    idx = 0
    for _, targets, image_paths in dataset_loader:
        print("For (non-resized) image:", image_paths[0])
        img = cv2.imread(image_paths[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)

        for target in targets:
            x1_list = []
            y1_list = []
            box_idx = 0  # Index for boxes and labels

            for bbox in target["boxes"]:
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox.tolist())
                    x1_list.append(x1)
                    y1_list.append(y1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            for label in target["labels"]:
                x1 = x1_list[box_idx]
                y1 = y1_list[box_idx]
                label = label.item()
                label_text = dataset_classes[label]

                if input_size[0] - x1 < 30 or input_size[1] - y1 < 30:
                    cv2.putText(img, str(label_text), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(img, str(label_text), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                box_idx += 1

        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        idx += 1
        if idx >= images_to_show:
            break

# Example usage
# show_samples(dataset_loader=dataset_loader, images_to_show=10, input_size=input_size)
