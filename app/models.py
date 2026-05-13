from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


VideoStatus = Literal["draft", "processing", "ready", "failed"]
JobStatus = Literal["queued", "running", "completed", "failed"]
StageStatus = Literal["pending", "running", "completed", "failed"]
ZoneKind = Literal["entrance", "checkout", "promo", "aisle", "rack", "custom"]


class Rect(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class Zone(BaseModel):
    id: str
    name: str
    kind: ZoneKind = "custom"
    color: str = "#6ee7b7"
    rect: Rect


class VideoAsset(BaseModel):
    id: str
    store_name: str
    camera_name: str
    recorded_at: str
    original_filename: str
    stored_filename: str
    uploaded_at: str
    file_path: str
    file_size_bytes: int
    status: VideoStatus = "draft"
    zones: list[Zone] = Field(default_factory=list)
    latest_job_id: str | None = None
    summary: str | None = None


class ProcessingStage(BaseModel):
    key: str
    label: str
    status: StageStatus = "pending"
    progress: int = 0
    updated_at: str


class AnalysisJob(BaseModel):
    id: str
    video_id: str
    engine: str
    status: JobStatus = "queued"
    progress: int = 0
    current_stage: str = "queued"
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    stages: list[ProcessingStage] = Field(default_factory=list)
    message: str = "Queued for analysis."


class SummaryMetrics(BaseModel):
    visitors: int
    estimated_buyers: int
    peak_occupancy: int
    avg_dwell_minutes: float
    avg_queue_minutes: float
    busiest_zone: str
    anomaly_count: int


class ZoneInsight(BaseModel):
    zone_id: str
    zone_name: str
    entries: int
    avg_dwell_minutes: float
    peak_occupancy: int
    congestion_score: float
    engagement_score: float


class HeatmapCell(BaseModel):
    row: int
    col: int
    intensity: float = Field(ge=0, le=1)


class JourneyPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    minute: int = Field(ge=0)


class PersonJourney(BaseModel):
    person_id: str
    label: str
    journey_type: str
    entered_at: str
    exited_at: str
    dwell_minutes: float
    zones_visited: list[str]
    path: list[JourneyPoint]
    thumbnail_hint: str


class Highlight(BaseModel):
    id: str
    title: str
    timestamp: str
    category: str
    severity: Literal["low", "medium", "high"]
    description: str


class Anomaly(BaseModel):
    id: str
    zone_name: str
    timestamp: str
    duration_minutes: float
    risk_level: Literal["medium", "high"]
    description: str


class InsightCard(BaseModel):
    title: str
    body: str


class ComparePanel(BaseModel):
    original_notes: list[str]
    annotated_notes: list[str]


class AnalysisResult(BaseModel):
    video_id: str
    engine: str
    generated_at: str
    summary_metrics: SummaryMetrics
    top_insights: list[InsightCard]
    zone_insights: list[ZoneInsight]
    heatmap: list[HeatmapCell]
    highlights: list[Highlight]
    journeys: list[PersonJourney]
    anomalies: list[Anomaly]
    compare_panel: ComparePanel
    recommended_questions: list[str]
    executive_summary: str


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    created_at: str


class AskVideoResponse(BaseModel):
    answer: str
    supporting_points: list[str]
    suggested_follow_ups: list[str]


class AppState(BaseModel):
    videos: dict[str, VideoAsset] = Field(default_factory=dict)
    jobs: dict[str, AnalysisJob] = Field(default_factory=dict)
    results: dict[str, AnalysisResult] = Field(default_factory=dict)
    conversations: dict[str, list[ConversationTurn]] = Field(default_factory=dict)


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
