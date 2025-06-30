import numpy as np
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def calculate_similarity(player1: Dict[str, Any], player2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two player feature dicts.
    Combines color, spatial, and texture features.
    """
    try:
        hist1 = player1["color_features"]["histogram"]
        hist2 = player2["color_features"]["histogram"]

        hist_corr = np.corrcoef(hist1, hist2)[0, 1]
        hist_sim = max(0, hist_corr) if not np.isnan(hist_corr) else 0

        size1 = player1["spatial_features"]["size"]
        size2 = player2["spatial_features"]["size"]
        size_diff = np.sqrt((size1[0] - size2[0]) ** 2 + (size1[1] - size2[1]) ** 2)
        size_sim = 1 / (1 + size_diff * 5)

        aspect_diff = abs(player1["spatial_features"]["aspect_ratio"] - player2["spatial_features"]["aspect_ratio"])
        aspect_sim = 1 / (1 + aspect_diff)

        texture_diff = np.linalg.norm(player1["texture_features"] - player2["texture_features"])
        texture_sim = 1 / (1 + texture_diff / 10)

        total_similarity = (
            0.35 * hist_sim
            + 0.25 * size_sim
            + 0.15 * aspect_sim
            + 0.15 * texture_sim
        )

        return max(0, min(1, total_similarity))

    except Exception as e:
        logger.warning(f"Similarity calculation failed: {e}")
        return 0.0 