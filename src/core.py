# src/core.py
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection import get_detector
from src.features import (
    extract_color_features,
    extract_spatial_features,
    extract_texture_features,
)
from src.gpu_manager import GPUManager
from src.similarity import calculate_similarity
from src.tracking import clean_old_tracks, update_tracks
from src.visualization import create_visualization_video, visualize_mapping_from_file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

"""
Core logic for the Soccer Player Cross-Mapping System.
Contains the PlayerMappingSystem class and related processing methods.
"""

class PlayerMappingSystem:
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the player mapping system

        Args:
            model_path: Path to fine-tuned YOLOv11 model. If not given, uses a default YOLO model
        """

        self.detector = get_detector(model_path, use_gpu)
        self.use_gpu = use_gpu
        self.player_tracks_broadcast = {}
        self.player_tracks_tacticam = {}
        self.mapping_results = {}
        self.frame_mappings = []
        self.processing_stats = {
            "frames_processed": 0,
            "players_detected_broadcast": 0,
            "players_detected_tacticam": 0,
            "successful_mappings": 0,
            "total_inference_time": 0.0,
            "average_inference_time": 0.0,
            "gpu_memory_peak": 0,
        }
        # Further increased threshold for even stricter matching
        self.similarity_threshold = 0.75

        self.gpu_manager = GPUManager()

        if self.use_gpu and self.gpu_manager.gpu_available:
            self.gpu_manager.optimize_gpu_settings()

    def detect_players_in_frame(
        self, frame: np.ndarray, frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        detections = self.detector.detect(frame)
        players = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    f"Skipping invalid crop: bbox={bbox} after clipping: ({x1},{y1},{
                        x2
                    },{y2})"
                )
                continue
            crop = frame[y1:y2, x1:x2]
            if (
                crop.size == 0
                or crop.shape[0] == 0
                or crop.shape[1] == 0
                or len(crop.shape) < 3
                or crop.shape[2] != 3
            ):
                logger.warning(
                    f"Skipping empty or invalid crop: bbox={bbox}, crop.shape={
                        crop.shape if crop is not None else None
                    }"
                )
                continue
            color_feats = extract_color_features(crop)
            spatial_feats = extract_spatial_features(bbox, frame.shape)
            texture_feats = extract_texture_features(crop)
            player = {
                "bbox": bbox,
                "confidence": det["confidence"],
                "class": det["class"],
                "color_features": color_feats,
                "spatial_features": spatial_feats,
                "texture_features": texture_feats,
            }
            players.append(player)
        return players

    def process_batch_frames(
        self, frames: List[np.ndarray], start_frame_idx: int = 0
    ) -> List[List[Dict]]:
        if not self.detector or not self.gpu_manager.gpu_available:
            return [
                self.detect_players_in_frame(frame, start_frame_idx + i)
                for i, frame in enumerate(frames)
            ]
        try:
            inference_start = time.time()
            valid_frames = []
            valid_indices = []
            skipped_frames = 0
            for idx, frame in enumerate(frames):
                # Robust validation for each frame
                if frame is None:
                    logger.warning(f"[Batch] Skipping None frame at batch index {idx}")
                    skipped_frames += 1
                    continue
                if not hasattr(frame, "size") or frame.size == 0:
                    logger.warning(f"[Batch] Skipping empty frame at batch index {idx}")
                    skipped_frames += 1
                    continue
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logger.warning(
                        f"[Batch] Skipping frame with invalid shape {
                            frame.shape
                        } at batch index {idx}"
                    )
                    skipped_frames += 1
                    continue
                if frame.dtype != np.uint8:
                    logger.warning(
                        f"[Batch] Skipping frame with dtype {
                            frame.dtype
                        } at batch index {idx}"
                    )
                    skipped_frames += 1
                    continue
                if not np.any(frame):
                    logger.warning(
                        f"[Batch] Skipping all-zero frame at batch index {idx}"
                    )
                    skipped_frames += 1
                    continue
                # Additional check: ensure frame has reasonable dimensions
                h, w = frame.shape[:2]
                if h < 10 or w < 10:
                    logger.warning(
                        f"[Batch] Skipping frame with too small dimensions {
                            frame.shape
                        } at batch index {idx}"
                    )
                    skipped_frames += 1
                    continue
                valid_frames.append(np.ascontiguousarray(frame))
                valid_indices.append(idx)
            if not valid_frames:
                logger.warning("All frames in batch are invalid, skipping batch.")
                return [[] for _ in frames]
            ref_shape = valid_frames[0].shape
            consistent_frames = [valid_frames[0]]
            consistent_indices = [valid_indices[0]]
            for frame, idx in zip(valid_frames[1:], valid_indices[1:]):
                if frame.shape == ref_shape:
                    consistent_frames.append(frame)
                    consistent_indices.append(idx)
                else:
                    logger.warning(
                        f"[Batch] Skipping frame at batch index {
                            idx
                        } due to shape mismatch: {frame.shape} != {ref_shape}"
                    )
                    skipped_frames += 1
            if not consistent_frames:
                logger.warning(
                    "No frames with consistent shape in batch, skipping batch."
                )
                return [[] for _ in frames]
            try:
                results_batch = self.detector.detect(consistent_frames)
            except Exception as e:
                logger.warning(
                    f"YOLO batch detection failed: {
                        e
                    }. Falling back to per-frame detection for this batch."
                )
                results_batch = [
                    self.detector.detect(frame) for frame in consistent_frames
                ]
            all_players = [[] for _ in frames]
            skipped_crops = 0
            for i, (frame, results) in zip(
                consistent_indices, zip(consistent_frames, results_batch)
            ):
                frame_players = []
                frame_idx = start_frame_idx + i
                for result in results:
                    bbox = result["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = frame.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(
                            f"Skipping invalid crop: bbox={bbox} after clipping: ({x1},{
                                y1
                            },{x2},{y2}) in frame {frame_idx}"
                        )
                        skipped_crops += 1
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if (
                        crop.size == 0
                        or crop.shape[0] == 0
                        or crop.shape[1] == 0
                        or len(crop.shape) < 3
                        or crop.shape[2] != 3
                    ):
                        logger.warning(
                            f"Skipping empty or invalid crop: bbox={bbox}, crop.shape={
                                crop.shape if crop is not None else None
                            } in frame {frame_idx}"
                        )
                        skipped_crops += 1
                        continue
                    color_feats = extract_color_features(crop)
                    spatial_feats = extract_spatial_features(bbox, frame.shape)
                    texture_feats = extract_texture_features(crop)
                    player = {
                        "bbox": bbox,
                        "confidence": result["confidence"],
                        "class": result["class"],
                        "color_features": color_feats,
                        "spatial_features": spatial_feats,
                        "texture_features": texture_feats,
                    }
                    frame_players.append(player)
                all_players[i] = frame_players
            inference_time = time.time() - inference_start
            self.processing_stats["total_inference_time"] += inference_time
            n_frames = len(consistent_frames)
            if n_frames > 0:
                self.processing_stats["average_inference_time"] = self.processing_stats[
                    "total_inference_time"
                ] / max(1, self.processing_stats["frames_processed"])
            return all_players
        except Exception as e:
            logger.error(f"Error in process_batch_frames: {e}")
            logger.error(traceback.format_exc())
            return [[] for _ in frames]

    def create_frame_mapping(
        self, players_broadcast: List[Dict], players_tacticam: List[Dict]
    ) -> Dict[str, str]:
        if not players_broadcast or not players_tacticam:
            return {}
        try:
            n_broadcast = len(players_broadcast)
            n_tacticam = len(players_tacticam)
            similarity_matrix = np.zeros((n_broadcast, n_tacticam))
            for i, p_broadcast in enumerate(players_broadcast):
                for j, p_tacticam in enumerate(players_tacticam):
                    similarity_matrix[i, j] = calculate_similarity(
                        p_broadcast, p_tacticam
                    )
            try:
                from scipy.optimize import linear_sum_assignment

                row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
            except ImportError:
                logger.warning("scipy not available, using greedy assignment")
                row_indices, col_indices = self._greedy_assignment(similarity_matrix)
            mapping = {}
            for i, j in zip(row_indices, col_indices):
                if (
                    i < n_broadcast
                    and j < n_tacticam
                    and similarity_matrix[i, j] > self.similarity_threshold
                ):
                    mapping[f"broadcast_{i}"] = f"tacticam_{j}"
            return mapping
        except Exception as e:
            logger.warning(f"Frame mapping failed: {e}")
            return {}

    def _greedy_assignment(
        self, similarity_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_rows, n_cols = similarity_matrix.shape
        row_indices = []
        col_indices = []
        used_cols = set()
        for i in range(n_rows):
            best_j = -1
            best_score = -1
            for j in range(n_cols):
                if j not in used_cols and similarity_matrix[i, j] > best_score:
                    best_score = similarity_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                row_indices.append(i)
                col_indices.append(best_j)
                used_cols.add(best_j)
        return np.array(row_indices), np.array(col_indices)

    def generate_global_mapping(self) -> Dict[str, Any]:
        mapping_confidence = {}
        global_mappings = defaultdict(lambda: defaultdict(int))
        for frame_mapping in self.frame_mappings:
            for broadcast_id, tacticam_id in frame_mapping.items():
                global_mappings[broadcast_id][tacticam_id] += 1
        final_mappings = {}
        for broadcast_id, tacticam_counts in global_mappings.items():
            if tacticam_counts:
                best_tacticam = max(tacticam_counts.items(), key=lambda x: x[1])
                confidence = best_tacticam[1] / sum(tacticam_counts.values())
                if confidence > 0.3:
                    final_mappings[broadcast_id] = best_tacticam[0]
                    mapping_confidence[broadcast_id] = confidence
        return {
            "mappings": final_mappings,
            "confidence": mapping_confidence,
            "statistics": {
                "total_frames_processed": len(self.frame_mappings),
                "unique_broadcast_players": len(global_mappings),
                "successful_mappings": len(final_mappings),
            },
        }

    def process_videos(
        self,
        broadcast_path: str,
        tacticam_path: str,
        output_path: str = "player_mapping_result.json",
        max_frames: int = 1000,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        logger.info("Starting GPU accelerated video processing")
        logger.info(f"Device: {self.gpu_manager.device_name}")
        logger.info(
            f"Batch size: {batch_size if self.gpu_manager.gpu_available else 1}"
        )
        if not os.path.exists(broadcast_path):
            logger.error(f"Broadcast video path: {broadcast_path}")
            return {}
        if not os.path.exists(tacticam_path):
            logger.error(f"Tactical camera video path: {tacticam_path}")
            return {}
        cap_broadcast = cv2.VideoCapture(broadcast_path)
        cap_tacticam = cv2.VideoCapture(tacticam_path)
        if not cap_broadcast.isOpened():
            logger.error("Failed to open broadcast video")
            return {}
        if not cap_tacticam.isOpened():
            logger.error("Failed to open tacticam video")
            return {}
        try:
            fps_broadcast = cap_broadcast.get(cv2.CAP_PROP_FPS)
            fps_tacticam = cap_tacticam.get(cv2.CAP_PROP_FPS)
            total_frames_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(
                f"Broadcast Video: {fps_broadcast:.2f} FPS, {total_frames_b} frames"
            )
            logger.info(
                f"Tacticam Video: {fps_tacticam:.2f} FPS, {total_frames_t} frames"
            )
            frame_count = 0
            skip_frames = 1
            processing_start_time = time.time()
            while frame_count < max_frames:
                ret_b, frame_b = cap_broadcast.read()
                ret_t, frame_t = cap_tacticam.read()
                if not ret_b or not ret_t:
                    break
                if frame_count % skip_frames == 0:
                    # Always do per-frame detection
                    broadcast_players = self.detect_players_in_frame(
                        frame_b, frame_count
                    )
                    tacticam_players = self.detect_players_in_frame(
                        frame_t, frame_count
                    )
                    update_tracks(
                        self.player_tracks_broadcast,
                        broadcast_players,
                        "broadcast",
                        frame_count,
                    )
                    update_tracks(
                        self.player_tracks_tacticam,
                        tacticam_players,
                        "tacticam",
                        frame_count,
                    )
                    frame_mapping = self.create_frame_mapping(
                        broadcast_players, tacticam_players
                    )
                    if frame_mapping:
                        self.frame_mappings.append(frame_mapping)
                        self.processing_stats["successful_mappings"] += len(
                            frame_mapping
                        )
                    self.processing_stats["players_detected_broadcast"] += len(
                        broadcast_players
                    )
                    self.processing_stats["players_detected_tacticam"] += len(
                        tacticam_players
                    )
                    self.processing_stats["frames_processed"] += 1
                    # Clean old tracks with higher max_age to reduce flicker
                    clean_old_tracks(
                        self.player_tracks_broadcast, frame_count, max_age=100
                    )
                    clean_old_tracks(
                        self.player_tracks_tacticam, frame_count, max_age=100
                    )
                frame_count += 1
            processing_time = time.time() - processing_start_time
            global_results = {
                "mappings": self.frame_mappings,
                "player_tracks_broadcast": self.player_tracks_broadcast,
                "player_tracks_tacticam": self.player_tracks_tacticam,
            }
            results = {
                "global_mapping": global_results,
                "processing_stats": self.processing_stats,
                "total_processing_time": processing_time,
                "videos_info": {
                    "broadcast": {
                        "fps": fps_broadcast,
                        "total_frames": total_frames_b,
                        "processed_frames": self.processing_stats["frames_processed"],
                    },
                    "tacticam": {
                        "fps": fps_tacticam,
                        "total_frames": total_frames_t,
                        "processed_frames": self.processing_stats["frames_processed"],
                    },
                },
                "frame_by_frame_mappings": [
                    {"frame": idx, "mapping": mapping}
                    for idx, mapping in enumerate(self.frame_mappings)
                ],
            }
            logger.info(f"Processing completed in {processing_time:.2f}s")
            logger.info(
                f"Frames processed: {self.processing_stats['frames_processed']}"
            )
            logger.info(
                f"Successful mappings: {self.processing_stats['successful_mappings']}"
            )
            logger.info(
                f"Average inference time: {
                    self.processing_stats['average_inference_time']:.4f}s"
            )
            if output_path:
                import json

                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {output_path}")
            return results
        except Exception as e:
            logger.error(f"Error processing videos: {e}")
            return {}
        finally:
            cap_broadcast.release()
            cap_tacticam.release()
            cv2.destroyAllWindows()

    def _save_results(self, output_path: str) -> None:
        results = {
            "final_mapping": self.mapping_results,
            "frame_by_frame_mappings": self.frame_mappings,
            "processing_statistics": self.processing_stats,
            "metadata": {
                "model_used": "YOLO" if self.detector else "OpenCV fallback",
                "total_frames_processed": len(self.frame_mappings),
                "average_players_per_frame_broadcast": (
                    self.processing_stats["players_detected_broadcast"]
                    / max(1, self.processing_stats["frames_processed"])
                ),
                "average_players_per_frame_tacticam": (
                    self.processing_stats["players_detected_tacticam"]
                    / max(1, self.processing_stats["frames_processed"])
                ),
            },
        }
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
