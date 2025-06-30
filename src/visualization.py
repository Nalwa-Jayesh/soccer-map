import cv2
import os
import logging
import json
from typing import List, Dict, Tuple, Any, Optional

"""
Visualization utilities for the Soccer Player Cross-Mapping System.
Provides functions to generate mapping visualizations from results.
"""

def create_visualization_video(
    broadcast_path: str,
    tacticam_path: str,
    mapping_results_path: str,
    output_video_path: str = "output/broadcast_with_mapping.mp4",
    max_frames: int = 1000,
    detect_players_fn=None,
) -> bool:
    """
    Create a visualization video showing player mappings between broadcast and tacticam views.
    Args:
        broadcast_path: Path to broadcast video.
        tacticam_path: Path to tacticam video.
        mapping_results_path: Path to mapping results JSON.
        output_video_path: Path to save the output video.
        max_frames: Maximum number of frames to process.
        detect_players_fn: Function to detect players in a frame (frame, frame_idx) -> List[Dict].
    Returns:
        True if successful, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating visualization video...")
    try:
        if not os.path.exists(mapping_results_path):
            logger.error(f"Mapping results file not found: {mapping_results_path}")
            return False
        if not os.path.exists(tacticam_path):
            logger.error(f"Tacticam video not found: {tacticam_path}")
            return False
        with open(mapping_results_path, "r") as f:
            results = json.load(f)
        frame_mappings = results.get("frame_by_frame_mappings", [])
        if not frame_mappings:
            logger.error("No frame mappings found in results")
            return False
        cap_broadcast = cv2.VideoCapture(broadcast_path)
        cap_tacticam = cv2.VideoCapture(tacticam_path)
        if not cap_broadcast.isOpened() or not cap_tacticam.isOpened():
            logger.error(
                f"Failed to open broadcast or tacticam video: {broadcast_path}, {tacticam_path}"
            )
            return False
        fps = cap_broadcast.get(cv2.CAP_PROP_FPS)
        width = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))
        if not out.isOpened():
            logger.error("Failed to create output video writer")
            cap_broadcast.release()
            cap_tacticam.release()
            return False
        frame_mapping_dict = {mapping_data["frame"]: mapping_data for mapping_data in frame_mappings}
        frame_count = 0
        processed_frames = 0
        logger.info(f"Processing {min(max_frames, total_frames)} frames...")
        while frame_count < max_frames:
            ret_b, frame_b = cap_broadcast.read()
            ret_t, frame_t = cap_tacticam.read()
            if not ret_b or not ret_t:
                break
            vis_b = frame_b.copy()
            vis_t = frame_t.copy()
            if frame_count in frame_mapping_dict:
                mapping_data = frame_mapping_dict[frame_count]
                frame_mappings_dict = mapping_data.get("mapping", {})
                # Detect players in both views
                players_b = detect_players_fn(frame_b, frame_count) if detect_players_fn else []
                players_t = detect_players_fn(frame_t, frame_count) if detect_players_fn else []
                # Build reverse mapping for tacticam
                reverse_mapping = {v: k for k, v in frame_mappings_dict.items()}
                # Draw green bboxes for mapped players
                for i, player in enumerate(players_b):
                    key = f"broadcast_{i}"
                    bbox = player["bbox"]
                    color = (
                        (0, 255, 0)
                        if key in frame_mappings_dict
                        else (128, 128, 128)
                    )
                    cv2.rectangle(
                        vis_b,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                for j, player in enumerate(players_t):
                    key = f"tacticam_{j}"
                    bbox = player["bbox"]
                    color = (
                        (0, 255, 0) if key in reverse_mapping else (128, 128, 128)
                    )
                    cv2.rectangle(
                        vis_t,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                processed_frames += 1
            vis = cv2.hconcat([vis_b, vis_t])
            vis = _add_frame_info_overlay(vis, frame_count, processed_frames)
            out.write(vis)
            if frame_count % 100 == 0:
                logger.info(
                    f"Processed {frame_count}/{min(max_frames, total_frames)} frames..."
                )
            frame_count += 1
        cap_broadcast.release()
        cap_tacticam.release()
        out.release()
        logger.info(
            f"Visualization video created successfully: {output_video_path}"
        )
        logger.info(f"Processed {processed_frames} frames with player mappings")
        return True
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def _add_frame_info_overlay(frame: Any, frame_num: int, processed_frames: int) -> Any:
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    cv2.putText(
        frame,
        f"Frame: {frame_num}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Mapped Frames: {processed_frames}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Player Mapping Visualization",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return frame

def visualize_mapping_from_file(
    broadcast_path: str,
    tacticam_path: str,
    mapping_results_path: str = "output/player_mapping_results.json",
    output_video_path: str = "output/broadcast_with_mapping.mp4",
    detect_players_fn=None,
) -> bool:
    """
    Wrapper to create visualization from mapping results file.
    """
    return create_visualization_video(
        broadcast_path=broadcast_path,
        tacticam_path=tacticam_path,
        mapping_results_path=mapping_results_path,
        output_video_path=output_video_path,
        detect_players_fn=detect_players_fn,
    ) 