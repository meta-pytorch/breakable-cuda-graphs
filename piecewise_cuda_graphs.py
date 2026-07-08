# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Piecewise CUDA graphs for PyTorch."""

from __future__ import annotations

import contextvars
import functools
import logging
import os
import threading
from types import TracebackType
from typing import Any, Callable, TypeGuard

import torch
import torch.cuda._gpu_trace as _gpu_trace

__version__ = "0.0.0"
__all__ = [
    "CUDAGraphSequence",
    "piecewise_graph",
    "no_graph",
    "force_no_graph",
    "is_in_piecewise_graph",
]

logger: logging.Logger = logging.getLogger(__name__)

_current_piecewise_graph_ctx: contextvars.ContextVar[piecewise_graph | None] = (
    contextvars.ContextVar("current_piecewise_graph_ctx", default=None)
)


def is_in_piecewise_graph() -> bool:
    """Return ``True`` while inside an active :class:`piecewise_graph` capture."""
    return _current_piecewise_graph_ctx.get() is not None


# Opt-in debug mode (read once at import). When on, the GPU-trace callbacks
# below track side-stream fork/join so an unjoined-stream error can name the
# offending stream id(s); it never changes capture behavior.
_DEBUG: bool = os.environ.get("PIECEWISE_CUDA_GRAPHS_DEBUG", "0") == "1"


def _on_event_record(event_id: int, stream_id: int) -> None:
    ctx = _current_piecewise_graph_ctx.get()
    if ctx is not None and _DEBUG:
        ctx._event_to_stream_id[event_id] = stream_id


def _on_event_wait(event_id: int, stream_id: int) -> None:
    ctx = _current_piecewise_graph_ctx.get()
    if ctx is None or not _DEBUG:
        return

    recording_stream_id = ctx._event_to_stream_id.get(event_id)
    if recording_stream_id is None:
        return

    recording_is_capturing = ctx._is_capturing_stream_id(recording_stream_id)
    waiting_is_capturing = ctx._is_capturing_stream_id(stream_id)

    if recording_is_capturing == waiting_is_capturing:
        return

    if recording_is_capturing:
        ctx._forked_stream_ids.add(stream_id)
    else:
        ctx._forked_stream_ids.discard(recording_stream_id)


_gpu_trace_initialized: bool = False
_gpu_trace_lock: threading.Lock = threading.Lock()


def _ensure_gpu_trace() -> None:
    global _gpu_trace_initialized
    if _gpu_trace_initialized:
        return
    with _gpu_trace_lock:
        if _gpu_trace_initialized:
            return
        torch._C._activate_gpu_trace()
        _gpu_trace.register_callback_for_event_record(_on_event_record)
        _gpu_trace.register_callback_for_event_wait(_on_event_wait)
        _gpu_trace_initialized = True


def _is_cuda_tensor(x: object) -> TypeGuard[torch.Tensor]:
    return isinstance(x, torch.Tensor) and x.is_cuda


def _make_replay_tensor_alias(x: object) -> object:
    if not _is_cuda_tensor(x):
        return x
    # TODO: Replace with torch._from_blob once pytorch/pytorch#185850 lands.
    us = x.untyped_storage()
    data_ptr = us.data_ptr()
    nbytes = us.nbytes()
    storage = torch._C._construct_storage_from_data_pointer(data_ptr, x.device, nbytes)
    metadata = {
        "data_ptr": data_ptr,
        "nbytes": nbytes,
        "device": x.device,
        "size": list(x.shape),
        "stride": list(x.stride()),
        "storage_offset": x.storage_offset(),
        "dtype": x.dtype,
    }
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)


def no_graph(
    fn: Callable[..., Any] | None = None, *, enable: bool = True
) -> Callable[..., Any]:
    """Run a function eagerly inside a :class:`piecewise_graph` capture.

    Calls to a decorated function end the current CUDA graph segment, execute the
    function normally, record it as an eager replay step, and then begin a new
    graph segment. Outside ``piecewise_graph`` capture, the wrapper calls the
    original function directly.

    Decorated functions must not return CUDA tensors. A CUDA tensor returned from
    an eager function cannot be safely reused across replays because its storage
    may move or be reused. Write CUDA outputs into pre-allocated buffers passed as
    arguments instead. Returning a CUDA tensor, including one nested in a tuple,
    list, or dict, raises :class:`RuntimeError` during capture. Scalars and CPU
    tensors are allowed.

    Can be used as ``@no_graph`` or ``@no_graph(enable=True)``. Passing
    ``enable=False`` leaves the function unchanged.

    Args:
        fn: Function to decorate. Leave as ``None`` when using the configured
            form, e.g. ``@no_graph(enable=...)``.
        enable: Whether to apply the eager-break wrapper. When ``False``,
            ``fn`` is returned unchanged.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not enable:
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ctx = _current_piecewise_graph_ctx.get()
            if ctx is None:
                return fn(*args, **kwargs)

            ctx._end_segment()

            captured_result = fn(*args, **kwargs)

            result_leaves, _ = torch.utils._pytree.tree_flatten(captured_result)
            if any(_is_cuda_tensor(leaf) for leaf in result_leaves):
                raise RuntimeError(
                    f"`{fn.__qualname__}` is decorated with `@no_graph` but "
                    f"returns one or more CUDA tensors. A CUDA tensor returned "
                    f"from an eager function cannot be safely reused across "
                    f"replays; write results into a pre-allocated buffer passed "
                    f"in as an argument instead."
                )

            # Store non-owning views of CUDA argument buffers. Eager segments are
            # kept for replay, so retaining the original tensors would extend
            # their lifetimes and can prevent the allocator from reusing those
            # buffers in later graph segments.
            replay_args = torch.utils._pytree.tree_map(_make_replay_tensor_alias, args)
            replay_kwargs = torch.utils._pytree.tree_map(
                _make_replay_tensor_alias, kwargs
            )

            ctx._insert_eager(fn, replay_args, replay_kwargs)
            ctx._begin_segment()

            return captured_result

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


@no_graph
def force_no_graph() -> None:
    pass


class _EagerSegment:
    """Eager replay step stored alongside captured CUDA graph segments.

    Holds a no-graph function plus non-owning argument-buffer views, and exposes
    the same ``replay()`` / ``reset()`` methods as :class:`torch.cuda.CUDAGraph`
    so :class:`CUDAGraphSequence` can replay all segments uniformly.
    """

    __slots__ = ("fn", "args", "kwargs")

    def __init__(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def replay(self) -> None:
        self.fn(*self.args, **self.kwargs)

    def reset(self) -> None:
        # No captured graph state to free; the args are non-owning pins.
        pass


class CUDAGraphSequence:
    """Capture output from :class:`piecewise_graph` that can be replayed.

    A sequence starts empty. Passing it to :class:`piecewise_graph` appends CUDA
    graph segments and eager ``@no_graph`` segments as capture progresses. Call
    :meth:`replay` to re-execute the captured sequence, or :meth:`reset` to drop
    all captured segments and reuse the object.

    Args:
        pool: Optional CUDA graph memory pool handle. Pass ``other.pool()`` to
            share a pool with another sequence; otherwise a pool is created
            lazily on first use.
    """

    def __init__(self, pool: Any | None = None) -> None:
        self._pool = pool
        self._segments: list[torch.cuda.CUDAGraph | _EagerSegment] = []

    def _append_graph(self) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        self._segments.append(g)
        return g

    def _append_eager(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self._segments.append(_EagerSegment(fn, args, kwargs))

    def pool(self) -> Any:
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        return self._pool

    def replay(self) -> None:
        for segment in self._segments:
            segment.replay()

    def reset(self) -> None:
        for segment in self._segments:
            segment.reset()
        self._segments.clear()


class piecewise_graph:
    """Capture CUDA work as multiple graph segments with eager breaks.

    This context manager behaves like :class:`torch.cuda.graph`, except calls to
    functions decorated with :func:`no_graph` run eagerly and split capture into
    separate CUDA graph segments. The resulting segments are appended to the
    provided :class:`CUDAGraphSequence`, which can then replay the full sequence.

    For concurrent captures from multiple threads with no-graph regions, use
    ``thread_local`` capture error mode and separate streams per thread.

    Args:
        cuda_graph_sequence: Sequence that receives captured graph segments and
            eager no-graph segments.
        stream: Optional side stream to use for capture. Matches the ``stream``
            argument of :class:`torch.cuda.graph`.
        capture_error_mode: CUDA stream-capture error mode forwarded to
            :class:`torch.cuda.graph`.

    Example:
        >>> @no_graph
        ... def copy_to_buffer(dst: torch.Tensor, src: torch.Tensor) -> None:
        ...     dst.copy_(src)
        ...
        >>> seq = CUDAGraphSequence()
        >>> static_input = torch.empty(5, device="cuda")
        >>> static_output = torch.empty_like(static_input)
        >>> with piecewise_graph(seq):
        ...     static_input.mul_(2)
        ...     copy_to_buffer(static_output, static_input)
        ...     static_output.add_(1)
        >>> seq.replay()
    """

    def __init__(
        self,
        cuda_graph_sequence: CUDAGraphSequence,
        stream: torch.cuda.Stream | None = None,
        capture_error_mode: str = "global",
    ) -> None:
        self._seq = cuda_graph_sequence
        self._stream = stream
        self._capture_error_mode = capture_error_mode
        self._graph_ctx: Any | None = None
        self._token: contextvars.Token[piecewise_graph | None] | None = None
        self._capturing_stream: torch.cuda.Stream | None = None
        self._capturing_stream_id: int | None = None
        self._forked_stream_ids: set[int] = set()
        self._event_to_stream_id: dict[int, int] = {}
        if _DEBUG:
            _ensure_gpu_trace()

    def _new_graph_ctx(self, g: torch.cuda.CUDAGraph) -> None:
        self._graph_ctx = torch.cuda.graph(
            g,
            pool=self._seq.pool(),
            stream=self._stream,
            capture_error_mode=self._capture_error_mode,
        )

    def _is_capturing_stream_id(self, stream_id: int) -> bool:
        return stream_id == self._capturing_stream_id

    def _insert_eager(
        self, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        assert not self._is_capturing()
        self._seq._append_eager(fn, args, kwargs)

    def _is_capturing(self) -> bool:
        return self._graph_ctx is not None

    def _begin_segment(self) -> None:
        g = self._seq._append_graph()
        self._new_graph_ctx(g)
        graph_ctx = self._graph_ctx
        assert graph_ctx is not None
        graph_ctx.__enter__()

    def _end_segment(self) -> None:
        try:
            graph_ctx = self._graph_ctx
            assert graph_ctx is not None
            graph_ctx.__exit__(None, None, None)
        except RuntimeError as e:
            # cudaErrorStreamCaptureUnjoined has no typed code; match its message
            # and re-raise anything else unchanged.
            text = str(e).lower()
            if "unjoined" not in text and "not joined" not in text:
                raise
            msg = (
                "piecewise_cuda_graphs: CUDA graph capture failed because a "
                "side stream was not joined back to the capturing stream before "
                "entering an @no_graph function. Join the side stream before "
                "the @no_graph call."
            )
            if _DEBUG and self._forked_stream_ids:
                msg += (
                    f" Unjoined side-stream id(s): {sorted(self._forked_stream_ids)}."
                )
            raise RuntimeError(msg) from e
        finally:
            self._graph_ctx = None
            if _DEBUG:
                self._event_to_stream_id.clear()

    def __enter__(self) -> piecewise_graph:
        if _current_piecewise_graph_ctx.get() is not None:
            raise RuntimeError("nested piecewise_graph captures are not supported")
        self._begin_segment()
        if _DEBUG:
            self._capturing_stream = torch.cuda.current_stream()
            self._capturing_stream_id = self._capturing_stream.cuda_stream
        self._token = _current_piecewise_graph_ctx.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        try:
            # _graph_ctx is None if an exception occurred during a no-graph
            # region (after _end_segment but before _begin_segment).
            graph_ctx = self._graph_ctx
            if graph_ctx is not None:
                if exc_type is not None:
                    # An exception is already unwinding; close best-effort and
                    # don't mask it with a cleanup error.
                    try:
                        graph_ctx.__exit__(exc_type, exc_val, exc_tb)
                    except RuntimeError as e:
                        logger.warning("CUDA graph cleanup failed: %s", e)
                else:
                    self._end_segment()
        finally:
            if _DEBUG:
                self._event_to_stream_id.clear()
            if self._token is not None:
                _current_piecewise_graph_ctx.reset(self._token)
        return False
