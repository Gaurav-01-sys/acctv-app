from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable

from app.models import (
    AnalysisResult,
    Anomaly,
    ComparePanel,
    HeatmapCell,
    Highlight,
    InsightCard,
    JourneyPoint,
    PersonJourney,
    Rect,
    SummaryMetrics,
    VideoAsset,
    Zone,
    ZoneInsight,
    utc_now,
)
from app.settings import Settings, get_settings



ProgressCallback = Callable[[str, int, str], None]


class AnalysisEngine(ABC):
    name: str

    @abstractmethod
    def analyze(self, video: VideoAsset, progress_callback: ProgressCallback | None = None) -> AnalysisResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Hungarian IoU Tracker: Direct box matching, no Kalman filter prediction
# Robust to frame skipping — matches on last-observed boxes only.
# Dependencies: numpy, scipy.optimize.linear_sum_assignment
# ---------------------------------------------------------------------------

class HungarianIoUTracker:
    """Simple multi-object tracker using IoU + Hungarian assignment.

    No Kalman filter — avoids prediction drift with frame skipping.
    Matches new detections against last-observed boxes directly.

    Interface: update(boxes) -> list[int] of track IDs (aligned with input).
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._next_id = 0
        self._tracks: dict[int, dict] = {}  # id -> {box, age}

    def update(self, boxes) -> list[int]:
        """Match detections to tracks, return aligned track IDs."""
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.ndim == 1 and boxes.shape[0] == 0:
            boxes = np.empty((0, 4))

        # Age all existing tracks
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1

        result_ids = [0] * len(boxes)

        if len(boxes) == 0:
            # Remove stale tracks
            self._tracks = {tid: t for tid, t in self._tracks.items()
                           if t["age"] <= self._max_age}
            return result_ids

        track_ids = list(self._tracks.keys())

        if track_ids:
            track_boxes = np.array([self._tracks[tid]["box"] for tid in track_ids])
            iou_matrix = self._iou_batch(boxes, track_boxes)
            cost_matrix = 1.0 - iou_matrix

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_dets = set()
            matched_trks = set()

            for det_idx, trk_idx in zip(row_indices, col_indices):
                if iou_matrix[det_idx, trk_idx] >= self._iou_threshold:
                    tid = track_ids[trk_idx]
                    self._tracks[tid]["box"] = boxes[det_idx]
                    self._tracks[tid]["age"] = 0
                    result_ids[det_idx] = tid
                    matched_dets.add(det_idx)
                    matched_trks.add(trk_idx)
        else:
            matched_dets = set()

        # Create new tracks for unmatched detections
        for det_idx in range(len(boxes)):
            if det_idx not in matched_dets:
                self._next_id += 1
                tid = self._next_id
                self._tracks[tid] = {"box": boxes[det_idx], "age": 0}
                result_ids[det_idx] = tid

        # Remove stale tracks
        self._tracks = {tid: t for tid, t in self._tracks.items()
                       if t["age"] <= self._max_age}

        return result_ids

    @staticmethod
    def _iou_batch(boxes_a, boxes_b):
        """Vectorized IoU between two sets of [x1, y1, x2, y2] boxes."""
        import numpy as np
        x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)


# ---------------------------------------------------------------------------
# ONNX-based YOLO Engine (no torch/ultralytics dependency)
# ---------------------------------------------------------------------------

class YoloStoreAnalyticsEngine(AnalysisEngine):
    name = "yolo26n-onnx-hungarian-iou"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._session = None

    def _get_model_path(self) -> str:
        """Download YOLO26n ONNX model from HuggingFace if not cached."""
        from huggingface_hub import hf_hub_download

        # Check if a local path was specified and exists
        local_path = Path(self.settings.yolo_model_path)
        if local_path.exists() and local_path.suffix == ".onnx":
            return str(local_path)

        # Download from HuggingFace
        # Use environment variables for customization
        repo_id = os.getenv("HF_YOLO_REPO", "qualcomm/YOLOv8-Detection-Quantized")
        filename = os.getenv("HF_YOLO_FILE", "YOLOv8-Detection-Quantized.onnx")

        cache_dir = Path(os.getenv("APP_DATA_DIR", "/tmp/aicctv-data")) / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir),
        )
        return model_file

    def _get_session(self):
        """Lazy-load the ONNX runtime session."""
        if self._session is None:
            import onnxruntime as ort
            model_path = self._get_model_path()
            self._session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
        return self._session

    def _preprocess(self, frame, np_module, target_size: int = 640):
        """Resize and normalize frame for YOLOv8 ONNX input."""
        import cv2
        h, w = frame.shape[:2]
        # Letterbox resize
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to target_size x target_size
        canvas = np_module.full((target_size, target_size, 3), 114, dtype=np_module.uint8)
        pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalize and transpose to NCHW
        blob = canvas.astype(np_module.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np_module.newaxis, ...]  # [1, 3, 640, 640]
        return blob, scale, pad_x, pad_y

    def _postprocess(self, output, scale, pad_x, pad_y, orig_w, orig_h, np_module,
                     conf_threshold: float = 0.5, iou_threshold: float = 0.5,
                     person_class: int = 0, target_size: int = 640):
        """Post-process YOLO26 end-to-end ONNX output [1, 300, 6] = [x1, y1, x2, y2, conf, class].
        NMS is already applied inside the model, so we only filter by class and confidence."""
        detections = output[0].squeeze()  # [300, 6]
        if detections.ndim == 1:
            detections = detections[np_module.newaxis, :]

        # Filter: person class AND confidence threshold
        mask = (detections[:, 5].astype(int) == person_class) & (detections[:, 4] >= conf_threshold)
        filtered = detections[mask]

        if len(filtered) == 0:
            return np_module.empty((0, 4)), np_module.empty(0)

        boxes_xyxy = filtered[:, :4].copy()
        scores = filtered[:, 4]

        # Scale from letterboxed 640x640 back to original image coordinates
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np_module.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np_module.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        return boxes_xyxy, scores

    def analyze(self, video: VideoAsset, progress_callback: ProgressCallback | None = None) -> AnalysisResult:
        try:
            import cv2
            import numpy as np
            from sklearn.cluster import DBSCAN
        except ImportError as exc:
            raise RuntimeError(
                "Core vision dependencies are missing (cv2, numpy, sklearn). Check requirements.txt."
            ) from exc

        zones = video.zones or self._default_zones()
        video_path = Path(video.file_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Uploaded video file not found: {video.file_path}")

        self._emit(progress_callback, "ingestion", 10, "Loading YOLO ONNX model from HuggingFace and reading video metadata.")

        # Load ONNX model session
        try:
            session = self._get_session()
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO ONNX model: {exc}") from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video.file_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration_s = round(total_frames / fps, 1) if total_frames else 0.0
        polygons = self._zone_polygons(zones, width, height)

        track_zone_dwell: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        track_paths: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
        track_first_seen: dict[int, float] = {}
        track_last_seen: dict[int, float] = {}
        track_current_zone: dict[int, str | None] = {}
        track_zone_entered_at: dict[int, float] = {}
        track_zones_visited: dict[int, list[str]] = defaultdict(list)
        track_frame_counts: dict[int, int] = defaultdict(int)

        zone_occupancy_over_time: dict[str, list[int]] = defaultdict(list)
        zone_centroids_over_time: dict[str, list[list[float]]] = defaultdict(list)
        unique_visitors: set[int] = set()
        loiter_flags: set[tuple[int, str]] = set()
        anomaly_log: list[dict[str, object]] = []
        heatmap = np.zeros((height, width), dtype=np.float32)

        processable_frames = max(1, math.ceil(max(total_frames, 1) / self.settings.frame_skip))
        frame_count = 0
        processed_frames = 0

        # Initialize tracker
        tracker = HungarianIoUTracker(iou_threshold=self.settings.yolo_iou_threshold, max_age=30)

        self._emit(progress_callback, "ingestion", 100, "Video metadata loaded.")
        self._emit(progress_callback, "detection", 5, "Running YOLO person detection via ONNX Runtime.")

        # Get ONNX input name
        input_name = session.get_inputs()[0].name

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if frame_count % self.settings.frame_skip != 0:
                continue

            processed_frames += 1
            timestamp_s = frame_count / fps

            # Run ONNX detection
            blob, scale, pad_x, pad_y = self._preprocess(frame, np)
            outputs = session.run(None, {input_name: blob})
            boxes, scores = self._postprocess(
                outputs, scale, pad_x, pad_y, width, height, np,
                conf_threshold=self.settings.yolo_conf_threshold,
                iou_threshold=self.settings.yolo_iou_threshold,
                person_class=self.settings.yolo_person_class,
            )

            # Update tracker
            ids = tracker.update(boxes)

            zone_counts: dict[str, int] = defaultdict(int)
            zone_centroids: dict[str, list[list[float]]] = defaultdict(list)
            active_track_ids: set[int] = set()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx = max(0, min(width - 1, int((x1 + x2) / 2)))
                cy = max(0, min(height - 1, int((y1 + y2) / 2)))
                foot_y = max(0, min(height - 1, int(y2)))

                active_track_ids.add(track_id)
                track_frame_counts[track_id] += 1

                if track_frame_counts[track_id] >= self.settings.min_track_frames:
                    unique_visitors.add(track_id)

                track_first_seen.setdefault(track_id, timestamp_s)
                track_last_seen[track_id] = timestamp_s

                if track_id not in unique_visitors:
                    continue

                if processed_frames % self.settings.path_sample_stride == 0:
                    track_paths[track_id].append((cx / width, cy / height, int(timestamp_s)))

                cv2.circle(
                    heatmap,
                    (cx, foot_y),
                    radius=self.settings.heatmap_radius,
                    color=1.0,
                    thickness=-1,
                )

                zone_id = self._zone_for_point(cx, cy, polygons, cv2, np)

                if track_id not in unique_visitors:
                    continue

                previous_zone = track_current_zone.get(track_id)
                previous_entry = track_zone_entered_at.get(track_id, timestamp_s)
                if previous_zone != zone_id:
                    if previous_zone is not None:
                        track_zone_dwell[track_id][previous_zone] += max(0.0, timestamp_s - previous_entry)
                    track_current_zone[track_id] = zone_id
                    track_zone_entered_at[track_id] = timestamp_s
                    if zone_id and zone_id not in track_zones_visited[track_id]:
                        track_zones_visited[track_id].append(zone_id)

                if zone_id:
                    zone_counts[zone_id] += 1
                    zone_centroids[zone_id].append([float(cx), float(cy)])
                    running_dwell = track_zone_dwell[track_id][zone_id] + max(
                        0.0,
                        timestamp_s - track_zone_entered_at.get(track_id, timestamp_s),
                    )
                    if running_dwell >= self.settings.loiter_threshold_s and (track_id, zone_id) not in loiter_flags:
                        loiter_flags.add((track_id, zone_id))
                        anomaly_log.append(
                            {
                                "track_id": track_id,
                                "zone_id": zone_id,
                                "timestamp_s": round(timestamp_s, 1),
                                "dwell_s": round(running_dwell, 1),
                                "type": "loitering",
                            }
                        )

            for zone in zones:
                zone_occupancy_over_time[zone.id].append(zone_counts[zone.id])
                if zone_centroids[zone.id]:
                    zone_centroids_over_time[zone.id].extend(zone_centroids[zone.id])

            if processed_frames % max(1, processable_frames // 12) == 0:
                progress = min(100, int((processed_frames / processable_frames) * 100))
                self._emit(progress_callback, "tracking", progress, f"Tracking people across frame {frame_count}.")

            for track_id in [track_id for track_id in list(track_current_zone.keys()) if track_id not in active_track_ids]:
                zone_id = track_current_zone.pop(track_id, None)
                entered_at = track_zone_entered_at.pop(track_id, None)
                if zone_id is not None and entered_at is not None:
                    track_zone_dwell[track_id][zone_id] += max(0.0, timestamp_s - entered_at)

        cap.release()
        final_timestamp_s = duration_s if duration_s > 0 else max(track_last_seen.values(), default=0.0)
        for track_id, zone_id in list(track_current_zone.items()):
            entered_at = track_zone_entered_at.get(track_id)
            if zone_id is not None and entered_at is not None:
                track_zone_dwell[track_id][zone_id] += max(0.0, final_timestamp_s - entered_at)

        self._emit(progress_callback, "tracking", 100, "Tracking completed.")
        self._emit(progress_callback, "aggregation", 25, "Aggregating zone, queue, and dwell metrics.")

        queue_analysis = {
            zone.id: self._estimate_queue_length(zone_centroids_over_time.get(zone.id, []), DBSCAN, np)
            for zone in zones
        }
        zone_by_id = {zone.id: zone for zone in zones}
        zone_stats = self._build_zone_stats(zones, track_zone_dwell, zone_occupancy_over_time, queue_analysis)
        zone_insights = [
            ZoneInsight(
                zone_id=zone.id,
                zone_name=zone.name,
                entries=int(zone_stats[zone.id]["visitors"]),
                avg_dwell_minutes=round(float(zone_stats[zone.id]["avg_dwell_s"]) / 60.0, 1),
                peak_occupancy=int(zone_stats[zone.id]["peak_occupancy"]),
                congestion_score=float(zone_stats[zone.id]["congestion_score"]),
                engagement_score=float(zone_stats[zone.id]["engagement_score"]),
            )
            for zone in zones
        ]

        total_dwell_minutes = [
            round(sum(zone_map.values()) / 60.0, 1)
            for zone_map in track_zone_dwell.values()
            if zone_map
        ]
        busiest_zone = max(zone_insights, key=lambda item: item.entries, default=None)
        busiest_zone_name = busiest_zone.zone_name if busiest_zone else (zones[0].name if zones else "Store Floor")
        checkout_zone_ids = [zone.id for zone in zones if zone.kind == "checkout" or "checkout" in zone.name.lower()]
        checkout_visitors = sum(
            1
            for zone_map in track_zone_dwell.values()
            if any(zone_map.get(zone_id, 0.0) > 0 for zone_id in checkout_zone_ids)
        )
        avg_queue_minutes = round(
            max((float(zone_stats[zone_id]["avg_dwell_s"]) / 60.0 for zone_id in checkout_zone_ids), default=0.0),
            1,
        )
        peak_occupancy = max(
            (max(series) for series in zone_occupancy_over_time.values() if series),
            default=0,
        )

        anomalies = self._build_anomalies(anomaly_log, queue_analysis, zone_by_id, final_timestamp_s)
        highlights = self._build_highlights(zone_insights, anomalies, zone_stats, zone_by_id, final_timestamp_s)
        journeys = self._build_journeys(
            track_paths,
            track_zone_dwell,
            track_zones_visited,
            track_first_seen,
            track_last_seen,
            zone_by_id,
        )
        top_insights = self._build_top_insights(zone_insights, anomalies, avg_queue_minutes)
        heatmap_cells = self._downsample_heatmap(heatmap, self.settings.heatmap_rows, self.settings.heatmap_cols, np)

        self._emit(progress_callback, "aggregation", 100, "Analytics aggregation completed.")
        self._emit(progress_callback, "insights", 85, "Generating highlights and executive summary.")

        executive_summary = (
            f"{video.store_name} / {video.camera_name} captured {len(unique_visitors)} tracked visitors. "
            f"{busiest_zone_name} carried the strongest activity, average dwell was "
            f"{round(sum(total_dwell_minutes) / len(total_dwell_minutes), 1) if total_dwell_minutes else 0.0} minutes, "
            f"checkout conversion is estimated at {checkout_visitors} visitors, and {len(anomalies)} anomaly events were flagged."
        )
        compare_panel = ComparePanel(
            original_notes=[
                f"Raw footage uploaded from {video.camera_name}",
                f"Video resolution {width}x{height} across {duration_s:.1f} seconds.",
                f"Store context: {video.store_name}",
            ],
            annotated_notes=[
                "YOLO26 detections were tracked across frames with BoT-SORT.",
                "Rectangle zones were converted into geometric hit-tests for every tracked centroid.",
                "Heatmap, dwell, queue, and anomaly metrics were generated from the tracked paths.",
            ],
        )

        result = AnalysisResult(
            video_id=video.id,
            engine=self.name,
            generated_at=utc_now(),
            summary_metrics=SummaryMetrics(
                visitors=len(unique_visitors),
                estimated_buyers=checkout_visitors,
                peak_occupancy=peak_occupancy,
                avg_dwell_minutes=round(sum(total_dwell_minutes) / len(total_dwell_minutes), 1) if total_dwell_minutes else 0.0,
                avg_queue_minutes=avg_queue_minutes,
                busiest_zone=busiest_zone_name,
                anomaly_count=len(anomalies),
            ),
            top_insights=top_insights,
            zone_insights=zone_insights,
            heatmap=heatmap_cells,
            highlights=highlights,
            journeys=journeys,
            anomalies=anomalies,
            compare_panel=compare_panel,
            recommended_questions=[
                "Which zone had the longest dwell time and what should the store manager do about it?",
                "Was checkout congestion worse than promo engagement?",
                "Which anomalies deserve review first?",
                "What layout or staffing change would likely reduce queue pressure?",
            ],
            executive_summary=executive_summary,
        )
        self._emit(progress_callback, "insights", 100, "Insight generation completed.")
        return result
    def _emit(self, callback: ProgressCallback | None, stage_key: str, progress: int, message: str) -> None:
        if callback:
            callback(stage_key, progress, message)

    def _default_zones(self) -> list[Zone]:
        return [
            Zone(id="entrance", name="Entrance", kind="entrance", color="#34d399", rect=Rect(x=0.05, y=0.08, width=0.2, height=0.26)),
            Zone(id="promo", name="Promo Wall", kind="promo", color="#f59e0b", rect=Rect(x=0.34, y=0.14, width=0.24, height=0.24)),
            Zone(id="aisle", name="Main Aisle", kind="aisle", color="#60a5fa", rect=Rect(x=0.18, y=0.45, width=0.48, height=0.2)),
            Zone(id="checkout", name="Checkout", kind="checkout", color="#f97316", rect=Rect(x=0.7, y=0.16, width=0.2, height=0.28)),
        ]

    def _zone_polygons(self, zones: list[Zone], width: int, height: int) -> dict[str, list[tuple[int, int]]]:
        polygons: dict[str, list[tuple[int, int]]] = {}
        for zone in zones:
            x1 = int(zone.rect.x * width)
            y1 = int(zone.rect.y * height)
            x2 = int((zone.rect.x + zone.rect.width) * width)
            y2 = int((zone.rect.y + zone.rect.height) * height)
            polygons[zone.id] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return polygons

    def _zone_for_point(
        self,
        cx: int,
        cy: int,
        polygons: dict[str, list[tuple[int, int]]],
        cv2_module: object,
        np_module: object,
    ) -> str | None:
        for zone_id, points in polygons.items():
            polygon = np_module.array(points, dtype=np_module.int32)
            if cv2_module.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
                return zone_id
        return None

    def _estimate_queue_length(self, centroids: list[list[float]], dbscan_cls: type, np_module: object) -> dict[str, int | float]:
        if len(centroids) < self.settings.queue_dbscan_min_samples:
            return {"queue_events": 0, "max_queue_length": 0, "avg_queue_length": 0.0}

        labels = dbscan_cls(
            eps=self.settings.queue_dbscan_eps,
            min_samples=self.settings.queue_dbscan_min_samples,
        ).fit_predict(np_module.array(centroids))

        cluster_sizes: list[int] = []
        for label in set(labels):
            if label == -1:
                continue
            size = int((labels == label).sum())
            if size >= self.settings.queue_min_size:
                cluster_sizes.append(size)

        if not cluster_sizes:
            return {"queue_events": 0, "max_queue_length": 0, "avg_queue_length": 0.0}
        return {
            "queue_events": len(cluster_sizes),
            "max_queue_length": max(cluster_sizes),
            "avg_queue_length": round(sum(cluster_sizes) / len(cluster_sizes), 1),
        }

    def _build_zone_stats(
        self,
        zones: list[Zone],
        track_zone_dwell: dict[int, dict[str, float]],
        zone_occupancy_over_time: dict[str, list[int]],
        queue_analysis: dict[str, dict[str, int | float]],
    ) -> dict[str, dict[str, float | int]]:
        max_peak = max((max(values) for values in zone_occupancy_over_time.values() if values), default=1)
        max_avg_dwell = max(
            (
                sum(zone_map.values()) / max(1, len(zone_map))
                for zone_map in track_zone_dwell.values()
                if zone_map
            ),
            default=1.0,
        )
        stats: dict[str, dict[str, float | int]] = {}

        for zone in zones:
            dwell_values = [
                zone_map[zone.id]
                for zone_map in track_zone_dwell.values()
                if zone_map.get(zone.id, 0.0) > 0
            ]
            occupancy = zone_occupancy_over_time.get(zone.id, [])
            peak_occupancy = max(occupancy, default=0)
            avg_dwell_s = round(sum(dwell_values) / len(dwell_values), 1) if dwell_values else 0.0
            queue_stats = queue_analysis.get(zone.id, {})
            congestion_score = min(
                0.99,
                round(((peak_occupancy / max_peak) * 0.6) + min(float(queue_stats.get("avg_queue_length", 0.0)) / 10.0, 0.39), 2),
            )
            engagement_score = min(0.99, round((avg_dwell_s / max_avg_dwell) if max_avg_dwell else 0.0, 2))
            if zone.kind == "promo":
                engagement_score = min(0.99, round(engagement_score + 0.08, 2))
            if zone.kind == "checkout":
                congestion_score = min(0.99, round(congestion_score + 0.1, 2))

            stats[zone.id] = {
                "visitors": len(dwell_values),
                "avg_dwell_s": avg_dwell_s,
                "max_dwell_s": round(max(dwell_values), 1) if dwell_values else 0.0,
                "peak_occupancy": peak_occupancy,
                "avg_occupancy": round(sum(occupancy) / len(occupancy), 2) if occupancy else 0.0,
                "max_queue_len": int(queue_stats.get("max_queue_length", 0)),
                "avg_queue_len": float(queue_stats.get("avg_queue_length", 0.0)),
                "queue_events": int(queue_stats.get("queue_events", 0)),
                "congestion_score": congestion_score,
                "engagement_score": engagement_score,
            }
        return stats

    def _downsample_heatmap(self, heatmap: object, rows: int, cols: int, np_module: object) -> list[HeatmapCell]:
        max_value = float(heatmap.max()) if heatmap.size else 0.0
        normalized = heatmap / max_value if max_value > 0 else np_module.zeros_like(heatmap)
        height, width = normalized.shape
        cells: list[HeatmapCell] = []
        for row in range(rows):
            for col in range(cols):
                y1 = int(row * height / rows)
                y2 = int((row + 1) * height / rows)
                x1 = int(col * width / cols)
                x2 = int((col + 1) * width / cols)
                region = normalized[y1:y2, x1:x2]
                intensity = float(region.mean()) if region.size else 0.0
                cells.append(HeatmapCell(row=row, col=col, intensity=round(min(1.0, intensity), 2)))
        return cells

    def _build_anomalies(
        self,
        anomaly_log: list[dict[str, object]],
        queue_analysis: dict[str, dict[str, int | float]],
        zone_by_id: dict[str, Zone],
        final_timestamp_s: float,
    ) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        for index, event in enumerate(anomaly_log[:5], start=1):
            zone = zone_by_id[str(event["zone_id"])]
            dwell_minutes = max(0.1, round(float(event["dwell_s"]) / 60.0, 1))
            anomalies.append(
                Anomaly(
                    id=f"anomaly-loiter-{index}",
                    zone_name=zone.name,
                    timestamp=self._format_timestamp(float(event["timestamp_s"])),
                    duration_minutes=dwell_minutes,
                    risk_level="high" if dwell_minutes >= 1.0 else "medium",
                    description=f"Track {event['track_id']} stayed in {zone.name} for {float(event['dwell_s']):.1f} seconds.",
                )
            )

        queue_zone_id = max(
            queue_analysis,
            key=lambda zone_id: float(queue_analysis[zone_id].get("max_queue_length", 0)),
            default=None,
        )
        if queue_zone_id and float(queue_analysis[queue_zone_id].get("max_queue_length", 0)) >= self.settings.queue_min_size:
            queue_zone = zone_by_id[queue_zone_id]
            anomalies.append(
                Anomaly(
                    id="anomaly-queue-1",
                    zone_name=queue_zone.name,
                    timestamp=self._format_timestamp(final_timestamp_s),
                    duration_minutes=max(
                        0.1,
                        round(float(queue_analysis[queue_zone_id].get("avg_queue_length", 0.0)) / 4.0, 1),
                    ),
                    risk_level="medium",
                    description=(
                        f"Queue-like clustering was detected in {queue_zone.name} with a maximum length of "
                        f"{int(queue_analysis[queue_zone_id].get('max_queue_length', 0))} people."
                    ),
                )
            )
        return anomalies[:6]

    def _build_highlights(
        self,
        zone_insights: list[ZoneInsight],
        anomalies: list[Anomaly],
        zone_stats: dict[str, dict[str, float | int]],
        zone_by_id: dict[str, Zone],
        final_timestamp_s: float,
    ) -> list[Highlight]:
        if not zone_insights:
            return []
        busiest = max(zone_insights, key=lambda item: item.entries)
        longest_dwell = max(zone_insights, key=lambda item: item.avg_dwell_minutes)
        queue_zone_id = max(zone_stats, key=lambda zone_id: int(zone_stats[zone_id]["max_queue_len"]), default=busiest.zone_id)
        queue_zone = zone_by_id[queue_zone_id]
        return [
            Highlight(
                id="highlight-1",
                title="Busiest traffic concentration",
                timestamp=self._format_timestamp(final_timestamp_s * 0.25),
                category="traffic",
                severity="medium",
                description=f"{busiest.zone_name} recorded the highest visitor count at {busiest.entries}.",
            ),
            Highlight(
                id="highlight-2",
                title="Longest dwell behaviour",
                timestamp=self._format_timestamp(final_timestamp_s * 0.5),
                category="behavior",
                severity="medium",
                description=f"{longest_dwell.zone_name} had the longest average dwell at {longest_dwell.avg_dwell_minutes} minutes.",
            ),
            Highlight(
                id="highlight-3",
                title="Queue pressure detected",
                timestamp=self._format_timestamp(final_timestamp_s * 0.75),
                category="queue",
                severity="high" if int(zone_stats[queue_zone_id]["max_queue_len"]) >= self.settings.queue_occupancy_max else "medium",
                description=(
                    f"{queue_zone.name} reached an estimated queue length of "
                    f"{int(zone_stats[queue_zone_id]['max_queue_len'])}."
                ),
            ),
            Highlight(
                id="highlight-4",
                title="Anomalies surfaced",
                timestamp=self._format_timestamp(final_timestamp_s),
                category="anomaly",
                severity="high" if anomalies else "low",
                description=anomalies[0].description if anomalies else "No major anomalies crossed the configured threshold.",
            ),
        ]

    def _build_top_insights(
        self,
        zones: list[ZoneInsight],
        anomalies: list[Anomaly],
        avg_queue_minutes: float,
    ) -> list[InsightCard]:
        if not zones:
            return []
        busiest = max(zones, key=lambda zone: zone.entries)
        strongest_engagement = max(zones, key=lambda zone: zone.engagement_score)
        highest_congestion = max(zones, key=lambda zone: zone.congestion_score)
        return [
            InsightCard(
                title="Busiest zone",
                body=f"{busiest.zone_name} handled the most traffic with {busiest.entries} tracked visits.",
            ),
            InsightCard(
                title="Checkout friction",
                body=f"Average queue dwell is estimated at {avg_queue_minutes} minutes for the checkout flow.",
            ),
            InsightCard(
                title="Engagement opportunity",
                body=f"{strongest_engagement.zone_name} showed the strongest engagement score in this session.",
            ),
            InsightCard(
                title="Operational watchpoint",
                body=anomalies[0].description if anomalies else f"{highest_congestion.zone_name} had the highest congestion score.",
            ),
        ]

    def _build_journeys(
        self,
        track_paths: dict[int, list[tuple[float, float, int]]],
        track_zone_dwell: dict[int, dict[str, float]],
        track_zones_visited: dict[int, list[str]],
        track_first_seen: dict[int, float],
        track_last_seen: dict[int, float],
        zone_by_id: dict[str, Zone],
    ) -> list[PersonJourney]:
        ranked_track_ids = sorted(
            track_zone_dwell,
            key=lambda track_id: sum(track_zone_dwell[track_id].values()),
            reverse=True,
        )[: self.settings.max_journeys]
        journeys: list[PersonJourney] = []
        for track_id in ranked_track_ids:
            zones_visited = [zone_by_id[zone_id].name for zone_id in track_zones_visited.get(track_id, []) if zone_id in zone_by_id]
            dwell_minutes = round(sum(track_zone_dwell[track_id].values()) / 60.0, 1)
            journeys.append(
                PersonJourney(
                    person_id=f"person-{track_id}",
                    label=self._journey_label(dwell_minutes, zones_visited),
                    journey_type=self._journey_type(dwell_minutes, zones_visited),
                    entered_at=self._format_timestamp(track_first_seen.get(track_id, 0.0)),
                    exited_at=self._format_timestamp(track_last_seen.get(track_id, 0.0)),
                    dwell_minutes=dwell_minutes,
                    zones_visited=zones_visited or ["Unzoned"],
                    path=[
                        JourneyPoint(x=point[0], y=point[1], minute=max(0, point[2] // 60))
                        for point in self._compact_path(track_paths.get(track_id, []))
                    ],
                    thumbnail_hint=self._journey_hint(zones_visited),
                )
            )
        return journeys

    def _compact_path(self, points: list[tuple[float, float, int]], limit: int = 24) -> list[tuple[float, float, int]]:
        if len(points) <= limit:
            return points
        step = max(1, len(points) // limit)
        return points[::step][:limit]

    def _journey_label(self, dwell_minutes: float, zones_visited: list[str]) -> str:
        if any("Checkout" in zone for zone in zones_visited):
            return "Checkout finisher"
        if dwell_minutes >= 4:
            return "Focused buyer"
        if len(zones_visited) >= 3:
            return "Exploratory browser"
        return "Quick browser"

    def _journey_type(self, dwell_minutes: float, zones_visited: list[str]) -> str:
        if any("Checkout" in zone for zone in zones_visited):
            return "decisive"
        if dwell_minutes >= 4:
            return "engaged"
        if len(zones_visited) >= 3:
            return "exploratory"
        return "purposeful"

    def _journey_hint(self, zones_visited: list[str]) -> str:
        if not zones_visited:
            return "Unzoned path"
        if len(zones_visited) == 1:
            return zones_visited[0]
        return f"{zones_visited[0]} to {zones_visited[-1]}"

    def _format_timestamp(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
