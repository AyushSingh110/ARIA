from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

_STORE_PATH = Path("data/experience_store.json")


class LocalExperienceStore:
    """File-backed JSON experience store for Phase 3.

    Swapped for MongoDB + ChromaDB in Phase 4 — the public interface
    (save / retrieve_similar / all_committed) stays identical.
    """

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self._records: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return []

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")

    def save(self, record: dict) -> str:
        record_id = str(uuid.uuid4())
        record["record_id"] = record_id
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        self._records.append(record)
        self._persist()
        return record_id

    def retrieve_similar(
        self,
        failure_class: str,
        task_class: str,
        k: int = 3,
        committed_only: bool = True,
    ) -> list[dict]:
        candidates = [
            r for r in self._records
            if r.get("failure_class") == failure_class
            and (not committed_only or r.get("committed", False))
        ]
        # Prefer same task class, then any
        same_class = [r for r in candidates if r.get("task_class") == task_class]
        ranked = same_class + [r for r in candidates if r not in same_class]
        # Sort by delta_score descending so the best refinements come first
        ranked.sort(key=lambda r: r.get("delta_score", 0.0), reverse=True)
        return ranked[:k]

    def all_committed(self) -> list[dict]:
        return [r for r in self._records if r.get("committed", False)]

    def count(self) -> int:
        return len(self._records)


@lru_cache(maxsize=1)
def get_store() -> LocalExperienceStore:
    return LocalExperienceStore()
