# Breakable CUDA Graphs - Design Document

## Overview

`breakable-cuda-graphs` is for workloads that should use CUDA graphs but contain
sections that are not CUDA-graph-compatible, or otherwise need to run eagerly.

The core idea is simple: instead of capturing one CUDA graph, we capture a linear
sequence of segments. Normal CUDA work is captured into `torch.cuda.CUDAGraph`
objects. Calls to `@no_graph` functions end the current segment, run eagerly,
record an eager replay step, and start the next graph segment.
`CUDAGraphSequence.replay()` replays that mixed sequence in order.

This work was inspired by
[SGLang PR #19102](https://github.com/sgl-project/sglang/pull/19102), which
introduced breakable CUDA graphs for SGLang's model executor.

## API

- **`CUDAGraphSequence`** - owns the captured sequence. It stores CUDA graph
  segments and eager segments, owns the shared CUDA graph memory pool, and
  exposes `replay()`, `reset()`, and `pool()`.
- **`breakable_graph`** - context manager that captures work into a
  `CUDAGraphSequence`. It forwards `stream` and `capture_error_mode` to
  `torch.cuda.graph` for each graph segment.
- **`@no_graph`** - marks a function as an eager break. Inside a
  `breakable_graph` capture, the wrapper ends the current graph segment, runs the
  function eagerly, records an eager segment, and starts the next graph segment.
  CUDA tensor returns are rejected; write CUDA outputs into argument buffers.
- **`force_no_graph()`** - `@no_graph`-decorated no-op for explicit split points.

## Architecture

**Dynamic segmentation.** The CUDA-graph-captured regions are dynamic, not
lexical. The user writes one `with breakable_graph(seq):` block, and each
`@no_graph` call splits the running capture wherever it occurs, including inside
helper functions or nested call stacks. Each stretch of CUDA-graph-compatible
execution between eager breaks is captured as its own graph segment.

**Segment sequence.** Instead of capturing into a single `torch.cuda.CUDAGraph`,
we capture those regions as a sequence of graph segments interleaved with eager
segments. The sequence starts empty; `breakable_graph` appends `CUDAGraph`
segments as `torch.cuda.graph` capture starts and eager segments when `@no_graph`
functions run. On `replay()`, graph segments replay and eager segments call their
stored functions.

**Context tracking.** A `contextvars.ContextVar` holds the active
`breakable_graph`. `@no_graph` wrappers use it to detect whether they are inside
a capture. Outside capture they call through directly; inside capture they end
the current graph segment, run eagerly, record an eager segment, and start the
next graph segment.

**Input/output contract.** CUDA graphs replay against fixed addresses, so eager
segments must see the same argument buffers every time. At capture we store
non-owning aliases to CUDA tensor inputs as the eager segment's replay arguments.
This pins inputs to their capture-time addresses without copying - it aliases the
same storage, it does not allocate.

Input pinning has no output equivalent: a returned CUDA tensor may be freshly
allocated at a different address each call, breaking the stable-address
constraint, and silently copying it into a static buffer every replay would hide
that cost. So we forbid it - a `@no_graph` function that returns one or more CUDA
tensors, including nested CUDA tensors, raises at capture time. Write outputs
into a pre-allocated buffer argument instead. Other returns, such as scalars or
CPU tensors, are allowed.

**Memory pool sharing.** The memory pool is owned by the `CUDAGraphSequence`
(lazily created on first use), and every segment captures into it - so the "all
segments share one pool" invariant is structural, not maintained per capture.
Pools can be shared across sequences via `CUDAGraphSequence(pool=other.pool())`.

**Side streams.** Ending a CUDA graph segment requires every participating side
stream to be joined back to the capturing stream. In breakable capture, this
matters at `@no_graph` boundaries because entering the eager function first ends
the current graph segment:

```
                    fork    no-graph FAILS
                    |       |
main  --[kernel A]--+-------+ !! FAILS: side stream not joined
                     \
side                  +--[kernel B]-- still dangling
```

Joining side streams is the caller's responsibility; `breakable-cuda-graphs` does
not auto-join them. Debug mode (`BREAKABLE_CUDA_GRAPHS_DEBUG=1`) helps identify
missing joins by tracking fork/join events and appending still-unjoined stream
id(s) to the error. It never joins streams or changes capture behavior.
