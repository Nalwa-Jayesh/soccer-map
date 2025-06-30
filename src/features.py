import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def extract_color_features(crop: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract color-based features from player crop."""
    # Robust check for valid color crop
    if (crop is None or crop.size == 0 or len(crop.shape) < 3 or crop.shape[0] == 0 or crop.shape[1] == 0 or crop.shape[2] != 3):
        mean_color = np.zeros(3)
        return {"histogram": np.zeros(96), "dominant_colors": np.tile(mean_color, 3)}
    try:
        hist_b = cv2.calcHist([crop], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([crop], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([crop], [2], None, [32], [0, 256])
        histogram = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        histogram = histogram / (histogram.sum() + 1e-7)
        pixels = crop.reshape(-1, 3).astype(np.float32)
        unique_colors, counts = np.unique(
            pixels.view(np.dtype((np.void, pixels.dtype.itemsize * pixels.shape[1]))),
            return_counts=True,
        )
        top_indices = np.argsort(counts)[-3:]
        dominant_colors = np.frombuffer(unique_colors[top_indices], dtype=pixels.dtype).reshape(-1, 3)
        if len(dominant_colors) < 3:
            dominant_colors = np.vstack([dominant_colors, np.zeros((3 - len(dominant_colors), 3))])
        return {"histogram": histogram, "dominant_colors": dominant_colors.flatten()}
    except Exception:
        mean_color = np.mean(crop, axis=(0, 1)) if crop.size > 0 else np.zeros(3)
        return {"histogram": np.zeros(96), "dominant_colors": np.tile(mean_color, 3)}

def extract_spatial_features(bbox: List[float], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    """Extract spatial features from bounding box."""
    x1, y1, x2, y2 = bbox
    width = max(1e-7, x2 - x1)
    height = max(1e-7, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    frame_height, frame_width = frame_shape[:2]
    norm_cx = center_x / frame_width
    norm_cy = center_y / frame_height
    norm_width = width / frame_width
    norm_height = height / frame_height
    aspect_ratio = width / height
    if width <= 0 or height <= 0:
        return {"center": (0.0, 0.0), "size": (0.0, 0.0), "aspect_ratio": 0.0}
    return {"center": (norm_cx, norm_cy), "size": (norm_width, norm_height), "aspect_ratio": aspect_ratio}

def extract_texture_features(crop: np.ndarray) -> np.ndarray:
    """Extract texture features using simple statistical measures."""
    # Robust check for valid color crop
    if (crop is None or crop.size == 0 or len(crop.shape) < 3 or crop.shape[0] == 0 or crop.shape[1] == 0 or crop.shape[2] != 3):
        return np.zeros(3)
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        grad_x = cv2.Sobel(gray, cv2.CV_64FC, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64FC, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(grad_mag)
        return np.array([mean_intensity, std_intensity, mean_gradient])
    except Exception:
        return np.zeros(3) 