from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseModel):
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_deployment_name: str | None = None

    yolo_model_path: str = "yolo26n.pt"
    yolo_person_class: int = 0
    yolo_conf_threshold: float = 0.5
    yolo_iou_threshold: float = 0.5
    tracker_config: str = "bytetrack.yaml"

    frame_skip: int = Field(default=5, ge=1)
    min_track_frames: int = Field(default=15, ge=1)
    loiter_threshold_s: float = Field(default=30.0, gt=0)
    queue_occupancy_max: int = Field(default=5, ge=1)
    queue_dbscan_eps: float = Field(default=80.0, gt=0)
    queue_dbscan_min_samples: int = Field(default=2, ge=1)
    queue_min_size: int = Field(default=3, ge=1)

    heatmap_rows: int = Field(default=6, ge=1)
    heatmap_cols: int = Field(default=6, ge=1)
    path_sample_stride: int = Field(default=8, ge=1)
    max_journeys: int = Field(default=5, ge=1)
    heatmap_radius: int = Field(default=15, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    deployment_name = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT")
    )
    model_path = os.getenv("YOLO_MODEL", "yolo26n.pt")
    resolved_model_path = Path(model_path)
    if not resolved_model_path.is_absolute():
        candidate = BASE_DIR / resolved_model_path
        if candidate.exists():
            resolved_model_path = candidate

    return Settings(
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        azure_openai_deployment_name=deployment_name,
        yolo_model_path=str(resolved_model_path),
        yolo_person_class=int(os.getenv("PERSON_CLASS", "0")),
        yolo_conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.5")),
        yolo_iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.5")),
        tracker_config=os.getenv("TRACKER_CONFIG", "bytetrack.yaml"),
        frame_skip=int(os.getenv("FRAME_SKIP", "5")),
        min_track_frames=int(os.getenv("MIN_TRACK_FRAMES", "15")),
        loiter_threshold_s=float(os.getenv("LOITER_THRESHOLD_S", "30")),
        queue_occupancy_max=int(os.getenv("QUEUE_OCCUPANCY_MAX", "5")),
        queue_dbscan_eps=float(os.getenv("QUEUE_DBSCAN_EPS", "80")),
        queue_dbscan_min_samples=int(os.getenv("QUEUE_DBSCAN_MINPTS", "2")),
        queue_min_size=int(os.getenv("QUEUE_MIN_SIZE", "3")),
        heatmap_rows=int(os.getenv("HEATMAP_ROWS", "6")),
        heatmap_cols=int(os.getenv("HEATMAP_COLS", "6")),
        path_sample_stride=int(os.getenv("PATH_SAMPLE_STRIDE", "8")),
        max_journeys=int(os.getenv("MAX_JOURNEYS", "5")),
        heatmap_radius=int(os.getenv("HEATMAP_RADIUS", "15")),
    )
