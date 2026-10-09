"""In-process event broker for WebSocket and SSE clients."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Tuple


class EventBroker:
    def __init__(self, max_queue: int = 200):
        self.max_queue = max_queue
        self._subscribers: Dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}
        self._lock = threading.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue)
        with self._lock:
            self._subscribers[queue] = asyncio.get_running_loop()
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.pop(queue, None)

    @staticmethod
    def _enqueue(queue: asyncio.Queue, event: Dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish(self, event: Dict[str, Any]) -> None:
        """Publish from a worker thread without touching an asyncio queue directly.

        Workers are ordinary threads, whereas WebSocket/SSE clients live on an
        asyncio event loop.  ``call_soon_threadsafe`` is the bridge between
        those two worlds.
        """

        with self._lock:
            subscribers = list(self._subscribers.items())
        for queue, loop in subscribers:
            try:
                loop.call_soon_threadsafe(self._enqueue, queue, event)
            except RuntimeError:
                # The subscriber's event loop may have shut down between the
                # snapshot and scheduling; it will be removed by its handler.
                continue
