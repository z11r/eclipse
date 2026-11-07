"""Bounded, sanitized event history; never stores OCR text or frame content.

Updates to an existing event receive a new sequence number so clients polling
`?after=...` receive the revised count. Events for different categories may
interleave without disrupting de-duplication.
"""
from collections import deque
from threading import Lock
from time import time
from uuid import uuid4
from .schemas import CatchEvent, CatchLog, Category, Source


class EventHistory:
    def __init__(self, capacity: int = 200, merge_window_s: float = 2.0):
        self._events: deque[CatchEvent] = deque(maxlen=capacity)
        self._last_seq = 0
        self._session_id = uuid4().hex
        self._lock = Lock()
        self._merge_window_s = merge_window_s

    def record(self, source: Source, category: Category, timestamp: float | None = None) -> CatchEvent:
        now = timestamp if timestamp is not None else time()
        with self._lock:
            # Find the most recent event of this source/category; other
            # categories can occur between successive detections.
            old = next((event for event in reversed(self._events)
                        if event.source == source and event.category == category), None)
            self._last_seq += 1
            if old is not None and 0 <= now - old.time <= self._merge_window_s:
                updated = CatchEvent(incident_id=old.incident_id, seq=self._last_seq, time=now,
                                     source=source, category=category,
                                     count=old.count + 1)
                # Replace in place so every category keeps its own rolling row.
                idx = next(i for i, event in enumerate(self._events) if event is old)
                self._events[idx] = updated
                return updated.model_copy()
            event = CatchEvent(incident_id=self._last_seq, seq=self._last_seq, time=now, source=source,
                               category=category, count=1)
            self._events.append(event)
            return event.model_copy()

    def since(self, after: int = 0, session_id: str | None = None) -> CatchLog:
        with self._lock:
            # A cursor belongs only to its history instance, even if the new
            # instance has already counted past that cursor after a restart.
            if session_id is not None and session_id != self._session_id:
                after = 0
            return CatchLog(session_id=self._session_id, events=[event.model_copy() for event in self._events
                                    if event.seq > after], last=self._last_seq)
