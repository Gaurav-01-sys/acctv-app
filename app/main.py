from __future__ import annotations

# Ensure vision packages are installed before importing anything else
from app.install_deps import ensure_packages  # noqa: E402

import os
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ask_video import answer_question
from app.engine import YoloStoreAnalyticsEngine
from app.models import (
    AnalysisJob,
    AppState,
    ConversationTurn,
    ProcessingStage,
    Rect,
    VideoAsset,
    Zone,
    utc_now,
)
from app.storage import JsonStore


BASE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("APP_DATA_DIR", str(BASE_DIR / "data"))).expanduser()
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", str(REPO_DIR / "frontend" / "dist"))).expanduser()
store = JsonStore(DATA_DIR)
engine = YoloStoreAnalyticsEngine()

app = FastAPI(title="CCTV Analytics API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST_DIR / "assets"), name="assets")


class ZonePayload(BaseModel):
    id: str | None = None
    name: str
    kind: str = "custom"
    color: str = "#6ee7b7"
    rect: Rect


class ZonesUpdatePayload(BaseModel):
    zones: list[ZonePayload] = Field(default_factory=list)


class AskVideoPayload(BaseModel):
    question: str = Field(min_length=1)


def default_stages() -> list[ProcessingStage]:
    now = utc_now()
    return [
        ProcessingStage(key="ingestion", label="Ingestion", updated_at=now),
        ProcessingStage(key="detection", label="People Detection", updated_at=now),
        ProcessingStage(key="tracking", label="Tracking", updated_at=now),
        ProcessingStage(key="aggregation", label="Zone Aggregation", updated_at=now),
        ProcessingStage(key="insights", label="Insight Generation", updated_at=now),
    ]


def apply_stage_update(job: AnalysisJob, stage_key: str, progress: int, message: str) -> AnalysisJob:
    now = utc_now()
    for stage in job.stages:
        if stage.key == stage_key:
            stage.status = "running" if progress < 100 else "completed"
            stage.progress = progress
            stage.updated_at = now
        elif stage.status == "running" and stage.key != stage_key:
            stage.status = "completed"
            stage.progress = 100
            stage.updated_at = now

    if progress == 100:
        current_index = next((index for index, stage in enumerate(job.stages) if stage.key == stage_key), -1)
        if 0 <= current_index + 1 < len(job.stages):
            next_stage = job.stages[current_index + 1]
            next_stage.status = "running"
            next_stage.progress = 5
            next_stage.updated_at = now
            job.current_stage = next_stage.label
        else:
            job.current_stage = "Completed"
    else:
        current = next((stage for stage in job.stages if stage.key == stage_key), None)
        job.current_stage = current.label if current else stage_key

    job.progress = progress
    job.message = message
    return job


def run_analysis_job(job_id: str, video_id: str) -> None:
    store.update(lambda state: _mark_job_running(state, job_id, video_id))
    try:
        video = store.load().videos[video_id]

        def progress_callback(stage_key: str, progress: int, message: str) -> None:
            store.update(
                lambda state, key=stage_key, pct=progress, msg=message: _update_job_stage(
                    state,
                    job_id,
                    key,
                    pct,
                    msg,
                )
            )

        result = engine.analyze(video, progress_callback=progress_callback)

        def finalize_success(state: AppState) -> None:
            job = state.jobs[job_id]
            current_video = state.videos[video_id]
            state.results[video_id] = result
            current_video.status = "ready"
            current_video.summary = result.executive_summary
            job.status = "completed"
            job.progress = 100
            job.current_stage = "Completed"
            job.completed_at = utc_now()
            job.message = "Analysis complete."
            for stage in job.stages:
                stage.status = "completed"
                stage.progress = 100
                stage.updated_at = utc_now()

            # Add system message to conversation
            conversation = state.conversations.setdefault(video_id, [])
            conversation.append(
                ConversationTurn(
                    role="system",
                    text="Analysis complete. You can now explore the insights or ask me questions about the footage!",
                    created_at=utc_now(),
                )
            )

        store.update(finalize_success)
    except Exception as exc:
        def finalize_failure(state: AppState) -> None:
            job = state.jobs[job_id]
            current_video = state.videos[video_id]
            current_video.status = "failed"
            job.status = "failed"
            job.completed_at = utc_now()
            job.message = str(exc)
            job.current_stage = "Failed"
            for stage in job.stages:
                if stage.status == "running":
                    stage.status = "failed"
                    stage.updated_at = utc_now()

        store.update(finalize_failure)


def _mark_job_running(state: AppState, job_id: str, video_id: str) -> None:
    job = state.jobs[job_id]
    video = state.videos[video_id]
    job.status = "running"
    job.started_at = utc_now()
    job.current_stage = "Ingestion"
    job.message = "Analysis started."
    job.stages[0].status = "running"
    job.stages[0].progress = 5
    job.stages[0].updated_at = utc_now()
    video.status = "processing"

    # Add system message to conversation
    conversation = state.conversations.setdefault(video_id, [])
    conversation.append(
        ConversationTurn(
            role="system",
            text="Analysis started. I am processing the video streams to extract insights.",
            created_at=utc_now(),
        )
    )


def _update_job_stage(state: AppState, job_id: str, stage_key: str, progress: int, message: str) -> None:
    job = state.jobs[job_id]
    apply_stage_update(job, stage_key, progress, message)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/api/debug/startup-log")
def startup_log():
    from pathlib import Path
    log_file = Path("/tmp/startup.log")
    if log_file.exists():
        return {"log": log_file.read_text(encoding="utf-8")}
    return {"log": "No startup log found"}



@app.get("/api/debug/packages")
def debug_packages() -> dict:
    """Check which vision packages are installed."""
    status = {}
    for pkg in ["cv2", "numpy", "sklearn", "ultralytics", "torch", "torchvision"]:
        try:
            mod = __import__(pkg)
            status[pkg] = getattr(mod, "__version__", "installed")
        except ImportError as e:
            status[pkg] = f"MISSING: {e}"
    
    import subprocess
    pip_list = subprocess.run(["pip", "list", "--format=columns"], capture_output=True, text=True)
    return {
        "packages": status,
        "pip_list_snippet": pip_list.stdout[:3000] if pip_list.returncode == 0 else pip_list.stderr[:1000]
    }



@app.get("/api/videos")
def list_videos() -> list[dict]:
    state = store.load()
    items = []
    for video in sorted(state.videos.values(), key=lambda item: item.uploaded_at, reverse=True):
        job = state.jobs.get(video.latest_job_id) if video.latest_job_id else None
        result = state.results.get(video.id)
        items.append(
            {
                "video": video.model_dump(mode="json"),
                "job": job.model_dump(mode="json") if job else None,
                "result": result.model_dump(mode="json") if result else None,
            }
        )
    return items


@app.post("/api/videos/upload")
async def upload_video(
    file: UploadFile = File(...),
    store_name: str = Form(...),
    camera_name: str = Form(...),
    recorded_at: str = Form(...),
) -> dict:
    video_id = f"video-{uuid.uuid4().hex[:8]}"
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    stored_filename = f"{video_id}{suffix}"
    destination = store.uploads_dir / stored_filename
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    video = VideoAsset(
        id=video_id,
        store_name=store_name,
        camera_name=camera_name,
        recorded_at=recorded_at,
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        uploaded_at=utc_now(),
        file_path=str(destination),
        file_size_bytes=destination.stat().st_size,
    )

    def create_video(state: AppState) -> VideoAsset:
        state.videos[video.id] = video
        return video

    created = store.update(create_video)
    return {"video": created.model_dump(mode="json"), "job": None, "result": None}


@app.get("/api/videos/{video_id}")
def get_video(video_id: str) -> dict:
    state = store.load()
    video = state.videos.get(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    job = state.jobs.get(video.latest_job_id) if video.latest_job_id else None
    result = state.results.get(video.id)
    conversation = state.conversations.get(video.id, [])
    return {
        "video": video.model_dump(mode="json"),
        "job": job.model_dump(mode="json") if job else None,
        "result": result.model_dump(mode="json") if result else None,
        "conversation": [turn.model_dump(mode="json") for turn in conversation],
    }


@app.put("/api/videos/{video_id}/zones")
def update_zones(video_id: str, payload: ZonesUpdatePayload) -> dict:
    def mutate(state: AppState) -> VideoAsset:
        video = state.videos.get(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        video.zones = [
            Zone(
                id=zone.id or f"zone-{index + 1}",
                name=zone.name,
                kind=zone.kind,  # type: ignore[arg-type]
                color=zone.color,
                rect=zone.rect,
            )
            for index, zone in enumerate(payload.zones)
        ]
        return video

    video = store.update(mutate)
    state = store.load()
    result = state.results.get(video.id)
    job = state.jobs.get(video.latest_job_id) if video.latest_job_id else None
    return {
        "video": video.model_dump(mode="json"),
        "job": job.model_dump(mode="json") if job else None,
        "result": result.model_dump(mode="json") if result else None,
    }


@app.post("/api/videos/{video_id}/analyze")
def analyze_video(video_id: str, background_tasks: BackgroundTasks) -> dict:
    now = utc_now()
    state = store.load()
    video = state.videos.get(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Prevent duplicate running jobs for the same video
    for job in state.jobs.values():
        if job.video_id == video_id and job.status in ("queued", "running"):
            return {
                "video": video.model_dump(mode="json"),
                "job": job.model_dump(mode="json"),
                "result": None,
                "message": "Analysis is already in progress for this video."
            }

    def create_job(state: AppState) -> tuple[VideoAsset, AnalysisJob]:
        video = state.videos.get(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        job = AnalysisJob(
            id=f"job-{uuid.uuid4().hex[:8]}",
            video_id=video_id,
            engine=engine.name,
            created_at=now,
            stages=default_stages(),
        )
        state.jobs[job.id] = job
        video.latest_job_id = job.id
        video.status = "processing"
        return video, job

    video, job = store.update(create_job)
    background_tasks.add_task(run_analysis_job, job.id, video_id)
    return {
        "video": video.model_dump(mode="json"),
        "job": job.model_dump(mode="json"),
        "result": None,
    }


@app.post("/api/videos/{video_id}/ask")
async def ask_video(video_id: str, payload: AskVideoPayload) -> dict:
    state = store.load()
    result = state.results.get(video_id)
    if not result:
        raise HTTPException(status_code=400, detail="Run analysis before asking questions.")
    history = state.conversations.get(video_id, [])
    response = await answer_question(result, payload.question, history)

    def mutate(state: AppState) -> dict:
        conversation = state.conversations.setdefault(video_id, [])
        conversation.append(ConversationTurn(role="user", text=payload.question, created_at=utc_now()))
        conversation.append(ConversationTurn(role="assistant", text=response.answer, created_at=utc_now()))
        return {
            "response": response.model_dump(mode="json"),
            "conversation": [turn.model_dump(mode="json") for turn in conversation],
        }

    return store.update(mutate)


@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str) -> dict:
    def mutate(state: AppState) -> dict:
        video = state.videos.pop(video_id, None)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Cleanup jobs
        state.jobs = {jid: j for jid, j in state.jobs.items() if j.video_id != video_id}
        # Cleanup results
        state.results.pop(video_id, None)
        # Cleanup conversations
        state.conversations.pop(video_id, None)

        # Cleanup physical file
        p = Path(video.file_path)
        if p.exists():
            p.unlink()

        return {"status": "deleted", "id": video_id}

    return store.update(mutate)


@app.delete("/api/videos")
def clear_all_videos() -> dict:
    def mutate(state: AppState) -> dict:
        # Physical cleanup of all uploads
        for video in state.videos.values():
            p = Path(video.file_path)
            if p.exists():
                p.unlink()

        # Reset state
        state.videos = {}
        state.jobs = {}
        state.results = {}
        state.conversations = {}
        return {"status": "cleared"}

    return store.update(mutate)


@app.get("/")
def serve_frontend_root() -> FileResponse:
    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run the frontend build first.")
    return FileResponse(index_file)


@app.get("/{full_path:path}")
def serve_frontend_app(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    requested_file = FRONTEND_DIST_DIR / full_path
    if requested_file.exists() and requested_file.is_file():
        return FileResponse(requested_file)

    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found. Run the frontend build first.")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
