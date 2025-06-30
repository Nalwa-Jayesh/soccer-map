from typing import Dict, List, Optional

import cv2
import numpy as np


class YOLODetector:
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        from ultralytics import YOLO

        self.model = YOLO(model_path) if model_path else YOLO("yolov8n.pt")
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def detect(self, frame_or_frames):
        """
        Detect players in a single frame or a batch of frames.
        Args:
            frame_or_frames: np.ndarray (H,W,3) or (N,H,W,3) or list of np.ndarray
        Returns:
            For single frame: List[Dict]
            For batch: List[List[Dict]]
        """
        # Handle batch input
        if isinstance(frame_or_frames, (list, tuple)) or (
            isinstance(frame_or_frames, np.ndarray) and frame_or_frames.ndim == 4
        ):
            # Convert list to np.ndarray if needed
            frames = (
                np.stack(frame_or_frames)
                if isinstance(frame_or_frames, (list, tuple))
                else frame_or_frames
            )
            results = self.model(frames, verbose=False)
            batch_detections = []
            for result in results:
                detections = []
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])
                        if conf > 0.85:
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                            detections.append(
                                {"bbox": bbox, "confidence": conf, "class": cls}
                            )
                batch_detections.append(detections)
            return batch_detections
        else:
            # Single frame
            results = self.model(frame_or_frames, verbose=False)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])
                        if conf > 0.9:
                            bbox = box.xyxy[0].cpu().numpy().tolist()
                            detections.append(
                                {"bbox": bbox, "confidence": conf, "class": cls}
                            )
            return detections


class OpenCVDetector:
    def __init__(self):
        self._background = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._background is None:
            self._background = blurred
            return []
        frame_diff = cv2.absdiff(self._background, blurred)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                if 0.3 < h / w < 3:
                    detections.append(
                        {
                            "bbox": [x, y, x + w, y + h],
                            "confidence": 0.7,
                            "class": 0,  # treat as person
                        }
                    )
        self._background = cv2.addWeighted(self._background, 0.9, blurred, 0.1, 0)
        return detections


def get_detector(model_path: Optional[str], use_gpu: bool):
    try:
        return YOLODetector(model_path, use_gpu)
    except ImportError:
        return OpenCVDetector()
