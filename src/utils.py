import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def ensure_dir(path: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")


def check_required_files(files: List[str]) -> bool:
    """Check if all required files exist. Returns True if all exist, else False and logs missing files."""
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        logger.error(f"Missing required files: {missing}")
        return False
    return True 