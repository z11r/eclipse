"""Numbers that stay the same size however long a stream runs."""

from __future__ import annotations

import math

_SMALLEST = 0.1  # below this, values share the first bin
_GROWTH = 1.03  # each bin is 3% wider than the last, so percentiles are within about 3%
_BINS = 512  # reaches about 360,000 from the smallest value


class Distribution:
    """Streaming percentiles in fixed memory, for values such as latency in ms."""

    def __init__(self) -> None:
        self._bins = [0] * _BINS
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self._bins[self._bin(value)] += 1

    def percentile(self, share: float) -> float:
        """The value below which this share of values fall, or 0 with none recorded."""
        if not self.count:
            return 0.0
        wanted = share * self.count
        seen = 0
        for index, count in enumerate(self._bins):
            seen += count
            if seen >= self.count:
                return self.maximum  # the last bin also holds everything past its edge
            if count and seen >= wanted:
                return min(_SMALLEST * _GROWTH ** (index + 1), self.maximum)
        return self.maximum

    def _bin(self, value: float) -> int:
        if value <= _SMALLEST:
            return 0
        return min(int(math.log(value / _SMALLEST, _GROWTH)), _BINS - 1)
