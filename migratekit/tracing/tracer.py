"""Writes trace events to a JSONL file, one event per line.

Thread-safe, because parallel workers will log to the same file.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

from .schema import TraceEvent, event_adapter


class Tracer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        line = event.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def read_events(path: str | Path) -> Iterator[TraceEvent]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield event_adapter.validate_json(line)