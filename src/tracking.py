import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def update_tracks(tracks: Dict, players: List[Dict], camera_type: str, frame_idx: int) -> None:
    """
    Update player tracks with new detections for a given frame.
    Args:
        tracks: Dictionary of current tracks
        players: List of detected player dicts
        camera_type: 'broadcast' or 'tacticam'
        frame_idx: Current frame index
    """
    for i, player in enumerate(players):
        track_id = f"{camera_type}_{i}"
        if track_id not in tracks:
            tracks[track_id] = {
                'track': [],
                'last_seen': frame_idx,
                'features': player
            }
        tracks[track_id]['track'].append((frame_idx, player['bbox']))
        tracks[track_id]['last_seen'] = frame_idx
        tracks[track_id]['features'] = player


def clean_old_tracks(tracks: Dict, current_frame: int, max_age: int = 50) -> None:
    """
    Remove tracks that haven't been updated for more than max_age frames.
    Args:
        tracks: Dictionary of current tracks
        current_frame: Current frame index
        max_age: Maximum allowed age for a track
    """
    to_delete = [tid for tid, t in tracks.items() if current_frame - t['last_seen'] > max_age]
    for tid in to_delete:
        logger.info(f"Removing stale track: {tid}")
        del tracks[tid] 