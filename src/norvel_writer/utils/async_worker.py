"""Qt/asyncio bridge: runs a persistent asyncio event loop on a QThread."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Coroutine, Optional, TypeVar

from PySide6.QtCore import QObject, QThread, Signal

log = logging.getLogger(__name__)

T = TypeVar("T")


class _LoopThread(QThread):
    """Background thread that owns an asyncio event loop."""

    def __init__(self) -> None:
        super().__init__()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = asyncio.Event()

    def run(self) -> None:  # type: ignore[override]
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        assert self._loop is not None, "Loop thread not started"
        return self._loop

    def stop(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self.wait()


class TaskSignals(QObject):
    """Signals emitted by async tasks back to the Qt main thread."""
    result = Signal(object)       # final result
    error = Signal(Exception)     # exception
    progress = Signal(int)        # 0-100
    chunk = Signal(str)           # streaming text chunk
    finished = Signal()


class AsyncWorker:
    """
    Singleton bridge between Qt and asyncio.

    Usage:
        worker = AsyncWorker.instance()
        worker.run(my_coroutine(), on_result=callback, on_error=err_callback)
    """

    _instance: Optional["AsyncWorker"] = None

    def __init__(self) -> None:
        self._thread = _LoopThread()
        self._thread.setDaemon(True)
        self._thread.start()

    @classmethod
    def instance(cls) -> "AsyncWorker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._thread.loop

    def run(
        self,
        coro: Coroutine[Any, Any, T],
        on_result: Optional[Callable[[T], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_progress: Optional[Callable[[int], None]] = None,
    ) -> "asyncio.Future[T]":
        """Schedule a coroutine. Callbacks are invoked on the main thread via signals."""
        signals = TaskSignals()
        if on_result:
            signals.result.connect(on_result)
        if on_error:
            signals.error.connect(on_error)
        if on_progress:
            signals.progress.connect(on_progress)

        async def _wrapped() -> T:
            try:
                result = await coro
                signals.result.emit(result)
                signals.finished.emit()
                return result
            except Exception as exc:
                log.exception("AsyncWorker task failed")
                signals.error.emit(exc)
                signals.finished.emit()
                raise

        future = asyncio.run_coroutine_threadsafe(_wrapped(), self.loop)
        return future  # type: ignore[return-value]

    def run_stream(
        self,
        agen: AsyncIterator[str],
        on_chunk: Callable[[str], None],
        on_done: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> "asyncio.Future[None]":
        """Consume an async generator, calling on_chunk for each yielded string."""
        signals = TaskSignals()
        signals.chunk.connect(on_chunk)
        if on_done:
            signals.finished.connect(on_done)
        if on_error:
            signals.error.connect(on_error)

        async def _consume() -> None:
            try:
                async for chunk in agen:
                    signals.chunk.emit(chunk)
                signals.finished.emit()
            except asyncio.CancelledError:
                signals.finished.emit()
            except Exception as exc:
                log.exception("Streaming task failed")
                signals.error.emit(exc)
                signals.finished.emit()

        future = asyncio.run_coroutine_threadsafe(_consume(), self.loop)
        return future  # type: ignore[return-value]

    def cancel_all(self) -> None:
        """Cancel all pending tasks on the loop."""
        async def _cancel() -> None:
            tasks = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
            for t in tasks:
                t.cancel()
        asyncio.run_coroutine_threadsafe(_cancel(), self.loop)

    def shutdown(self) -> None:
        self.cancel_all()
        self._thread.stop()
