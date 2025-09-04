
import numpy as np


def polygon_to_bbox(coords):
    """
    Args: coords - coords (scaled) of our dataset, from the .txt files
    Returns:
          x_min, y_min, x_max, y_max - coords of the bbox, abs values
    """

    x_coords = coords[0::2]
    y_coords = coords[1::2]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
  
    return [x_min, y_min, x_max, y_max]



def clamp(value, min_val, max_val):
    """Ensure the value stays within the specified bounds."""
    return max(min(value, max_val), min_val)
