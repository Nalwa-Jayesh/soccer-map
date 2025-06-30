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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    def _calculate_mapping_confidence(self, mapping: Dict[str, str]) -> float:
        if not mapping:
            return 0.0
        return min(1.0, len(mapping) / 10.0)

    def _create_final_mapping(self) -> Dict[str, Any]:
        logger.info("Creating final player mapping...")
        player_associations = defaultdict(list)
        for frame_data in self.frame_mappings:
            for broadcast_player, tacticam_player in frame_data["mapping"].items():
                player_associations[broadcast_player].append(tacticam_player)
        final_mapping = {}
        mapping_confidence = {}
        for broadcast_player, tacticam_players in player_associations.items():
            if tacticam_players:
                unique_players, counts = np.unique(tacticam_players, return_counts=True)
                most_common_idx = np.argmax(counts)
                most_common_player = unique_players[most_common_idx]
                confidence = counts[most_common_idx] / len(tacticam_players)
                final_mapping[broadcast_player] = most_common_player
                mapping_confidence[broadcast_player] = confidence
        return {
            "player_mapping": final_mapping,
            "mapping_confidence": mapping_confidence,
            "statistics": self.processing_stats,
            "total_unique_broadcast_players": len(final_mapping),
            "total_unique_tacticam_players": len(set(final_mapping.values())),
        }

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

    def create_visualization_video(
        self,
        broadcast_path: str,
        mapping_results_path: str,
        output_video_path: str = "output/broadcast_with_mapping.mp4",
        max_frames: int = 1000,
        tacticam_path: str = "tacticam.mp4",
    ) -> bool:
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
            final_mapping = results.get("final_mapping", {}).get("player_mapping", {})
            if not frame_mappings:
                logger.error("No frame mappings found in results")
                return False
            cap_broadcast = cv2.VideoCapture(broadcast_path)
            cap_tacticam = cv2.VideoCapture(tacticam_path)
            if not cap_broadcast.isOpened() or not cap_tacticam.isOpened():
                logger.error(
                    f"Failed to open broadcast or tacticam video: {broadcast_path}, {
                        tacticam_path
                    }"
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
            frame_mapping_dict = {}
            for mapping_data in frame_mappings:
                frame_num = mapping_data["frame"]
                frame_mapping_dict[frame_num] = mapping_data
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
                    # Get mapping for this frame
                    frame_mappings_dict = mapping_data.get("mapping", {})
                    # Detect players in both views
                    players_b = self.detect_players_in_frame(frame_b, frame_count)
                    players_t = self.detect_players_in_frame(frame_t, frame_count)
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
                # Side-by-side view
                vis = cv2.hconcat([vis_b, vis_t])
                vis = self._add_frame_info_overlay(vis, frame_count, processed_frames)
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
            logger.error(traceback.format_exc())
            return False

    def _create_persistent_player_ids(
        self, final_mapping: Dict[str, str]
    ) -> Dict[str, int]:
        player_id_map = {}
        current_id = 1
        sorted_broadcast_players = sorted(final_mapping.keys())
        for broadcast_player in sorted_broadcast_players:
            player_id_map[broadcast_player] = current_id
            current_id += 1
        return player_id_map

    def _apply_player_visualizations(
        self,
        frame: np.ndarray,
        players: List[Dict],
        mapping_data: Dict,
        player_id_map: Dict[str, int],
        colors: List[Tuple[int, int, int]],
        frame_num: int,
    ) -> np.ndarray:
        vis_frame = frame.copy()
        frame_mappings = mapping_data.get("mapping", {})
        # Assign a unique color for each mapping pair
        mapping_colors = {}
        color_palette = colors
        color_count = len(color_palette)
        # Assign a color to each mapped tacticam player
        for i, (broadcast_key, tacticam_player) in enumerate(frame_mappings.items()):
            # Use the tacticam_player index (if present) to assign color
            try:
                idx = int(tacticam_player.split("_")[-1])
                mapping_colors[broadcast_key] = color_palette[idx % color_count]
            except Exception:
                mapping_colors[broadcast_key] = (128, 128, 128)  # fallback gray
        for i, player in enumerate(players):
            bbox = player["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            broadcast_key = f"broadcast_{i}"
            player_id = player_id_map.get(broadcast_key, -1)
            tacticam_player = frame_mappings.get(broadcast_key, "No Match")
            # Use the mapping color if available, else gray
            color = mapping_colors.get(broadcast_key, (128, 128, 128))
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            if player_id > 0:
                id_text = f"P{player_id}"
                cv2.putText(
                    vis_frame,
                    id_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            confidence = player.get("confidence", 0)
            conf_text = f"{confidence:.2f}"
            cv2.putText(
                vis_frame,
                conf_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            mapping_text = tacticam_player.replace("tacticam_", "T")
            cv2.putText(
                vis_frame,
                mapping_text,
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return vis_frame

    def _add_frame_info_overlay(
        self, frame: np.ndarray, frame_num: int, processed_frames: int
    ) -> np.ndarray:
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
        self,
        broadcast_path: str,
        mapping_results_path: str = "output/player_mapping_results.json",
        output_video_path: str = "output/broadcast_with_mapping.mp4",
    ) -> bool:
        logger.info("=== Creating Player Mapping Visualization ===")
        if not os.path.exists(broadcast_path):
            logger.error(f"Broadcast video not found: {broadcast_path}")
            return False
        if not os.path.exists(mapping_results_path):
            logger.error(f"Mapping results not found: {mapping_results_path}")
            return False
        success = self.create_visualization_video(
            broadcast_path=broadcast_path,
            mapping_results_path=mapping_results_path,
            output_video_path=output_video_path,
            max_frames=1000,
        )
        if success:
            logger.info("✅ Visualization completed successfully!")
            logger.info(f"📹 Output video: {output_video_path}")
            logger.info("\nVisualization Features:")
            logger.info("- Colored bounding boxes for each player")
            logger.info("- Persistent player IDs (P1, P2, etc.)")
            logger.info("- Detection confidence scores")
            logger.info("- Tacticam mapping information")
            logger.info("- Frame counter and statistics")
        else:
            logger.error("❌ Visualization creation failed")
        return success
