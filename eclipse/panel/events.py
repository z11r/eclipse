"""Bounded, sanitized event history. No OCR text is accepted or stored."""
from collections import deque
from threading import Lock
from time import time
from .schemas import CatchEvent, CatchLog, Category, Source


class EventHistory:
    def __init__(self, capacity: int = 200, merge_window_s: float = 2.0):
        self._events: deque[CatchEvent] = deque(maxlen=capacity)
        self._last_seq = 0
        self._lock = Lock()
        self._merge_window_s = merge_window_s

    def record(self, source: Source, category: Category, timestamp: float | None = None) -> CatchEvent:
        now = timestamp if timestamp is not None else time()
        with self._lock:
            if self._events:
                current = self._events[-1]
                if (current.source == source and current.category == category
                        and 0 <= now - current.time <= self._merge_window_s):
                    # Give the updated event a NEW seq so clients using ?after see updates.
                    self._last_seq += 1
                    updated = CatchEvent(seq=self._last_seq, time=now, source=source,
                                         category=category, count=current.count + 1)
                    self._events[-1] = updated
                    return updated.model_copy()
            self._last_seq += 1
            event = CatchEvent(seq=self._last_seq, time=now, source=source,
                               category=category, count=1)
            self._events.append(event)
            return event.model_copy()

    def since(self, after: int = 0) -> CatchLog:
        with self._lock:
            return CatchLog(events=[e.model_copy() for e in self._events if e.seq > after],
                            last=self._last_seq)
