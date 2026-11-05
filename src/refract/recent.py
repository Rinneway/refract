"""A mapping that keeps only the most recent entries — for in-memory runs and API jobs of a long-running server."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Recent(OrderedDict, Generic[K, V]):
    """A dict that forgets the oldest entries beyond `limit`. Reading an entry does not refresh it: an insight or a
    job lives as long as it is among the `limit` newest ones."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = max(1, limit)
        self._lock = threading.Lock()

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            if key in self:
                super().__delitem__(key)
            super().__setitem__(key, value)
            while len(self) > self.limit:
                self.popitem(last=False)
