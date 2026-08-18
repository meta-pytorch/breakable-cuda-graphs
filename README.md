# Breakable CUDA Graphs

Breakable CUDA Graphs extends the standard PyTorch CUDA graph workflow.

With `torch.cuda.graph`, one context manager captures one CUDA graph. That works
well when the whole region is capture-compatible. `breakable-cuda-graphs` keeps
the same capture/replay shape, but lets one `with breakable_graph(...)` block
produce a sequence of CUDA graph segments separated by explicit eager breaks. If
no eager breaks occur, the block is captured as a single graph segment.

Use this when most of a workload should run under CUDA graphs, but some sections
are not CUDA-graph-compatible or otherwise need to run eagerly. Mark those
sections with `@no_graph`, and make their CUDA inputs/outputs obey the
constraints below.

## Quick start

Install from source into an environment with PyTorch and CUDA support:

```bash
git clone https://github.com/meta-pytorch/breakable-cuda-graphs.git
cd breakable-cuda-graphs
pip install -e .
```

The capture flow is the same as regular CUDA graphs: allocate static buffers,
warm up on a side stream, capture, then replay by updating the static inputs.
The only addition is `@no_graph`, which marks functions that should run eagerly
between captured graph segments.

```python
import torch
from breakable_cuda_graphs import CUDAGraphSequence, no_graph, breakable_graph

@no_graph
def dynamic_scale(x: torch.Tensor) -> None:
    # Cannot be captured: reads a value back to the CPU.
    if x.sum().item() > 0:
        x.clamp_(min=0)

# Pre-allocate static buffers.
static_input = torch.empty(1024, device="cuda")
result = torch.empty(1024, device="cuda")

def workload(src: torch.Tensor, dst: torch.Tensor) -> None:
    dst.copy_(src * 2)
    dynamic_scale(dst)  # ends the current graph segment and runs eagerly
    dst.add_(1.0)

# Warm up on a side stream, as required by CUDA graphs.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_input.fill_(1.0)
        workload(static_input, result)
torch.cuda.current_stream().wait_stream(s)

# Capture.
seq = CUDAGraphSequence()
with breakable_graph(seq):
    workload(static_input, result)

# Replay with new data by overwriting the static input buffer.
static_input.fill_(5.0)
seq.replay()
```

## Constraints

- `@no_graph` functions must not return CUDA tensors. Write CUDA outputs into
  pre-allocated buffers passed as arguments.
- Side streams must be joined back to the capturing stream before entering an
  `@no_graph` function or leaving the `breakable_graph` context. Set
  `BREAKABLE_CUDA_GRAPHS_DEBUG=1` to add still-unjoined stream id(s) to the
  resulting error.
- Usual CUDA graph constraints still apply: replay uses the same tensor addresses
  captured during warmup/capture.

## Additional usage

### Explicit split points

`force_no_graph()` inserts a graph break with no eager work. This can be useful
for debugging or isolating capture regions.

```python
from breakable_cuda_graphs import CUDAGraphSequence, force_no_graph, breakable_graph

seq = CUDAGraphSequence()
with breakable_graph(seq):
    a = step1(x)
    force_no_graph()
    b = step2(a)
```

### Sharing memory pools

All graph segments within a sequence share the same CUDA graph memory pool. You
can also share pools across sequences:

```python
seq1 = CUDAGraphSequence()
with breakable_graph(seq1):
    workload_a(buf_a, src_a)

seq2 = CUDAGraphSequence(pool=seq1.pool())
with breakable_graph(seq2):
    workload_b(buf_b, src_b)
```

## Reference

- **`CUDAGraphSequence(pool=None)`**: captured graph/eager segment sequence.
  Methods: `replay()`, `reset()`, `pool()`.
- **`breakable_graph(seq, stream=None, capture_error_mode="global")`**: capture
  context, analogous to `torch.cuda.graph`.
- **`@no_graph` / `@no_graph(enable=...)`**: mark functions that run eagerly
  inside `breakable_graph`.
- **`force_no_graph()`**: explicit split point with no eager work.

For implementation details, see [DESIGN.md](DESIGN.md).

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
