from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Callable, TypeVar

from app.models import AppState

T = TypeVar("T")


class JsonStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.base_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.base_dir / "store.json"
        self._lock = Lock()
        if not self.file_path.exists():
            self.save(AppState())

    def load(self) -> AppState:
        with self._lock:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
        return AppState.model_validate(payload)

    def save(self, state: AppState) -> None:
        with self._lock:
            self.file_path.write_text(
                json.dumps(state.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

    def update(self, callback: Callable[[AppState], T]) -> T:
        with self._lock:
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            state = AppState.model_validate(payload)
            result = callback(state)
            self.file_path.write_text(
                json.dumps(state.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            return result
