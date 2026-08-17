# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for breakable_cuda_graphs."""

import copy
import gc
import threading
import unittest
import weakref
from unittest.mock import patch

import breakable_cuda_graphs as bcg
import pytest
import torch
from absl.testing import parameterized
from breakable_cuda_graphs import (
    CUDAGraphSequence,
    breakable_graph,
    force_no_graph,
    is_in_breakable_graph,
    no_graph,
)


class ForceCUDAGraphGC:
    """Mixin that enables force_cudagraph_gc for the duration of each test.

    Multi-stream tests leave CUDAGraph objects with side stream references.
    When these are GC'd lazily, their stale capture state causes subsequent
    captures to fail. This mixin enables PyTorch's gc.collect() before each
    capture to ensure deterministic cleanup.
    """

    def setUp(self):
        super().setUp()
        self._old_force_cudagraph_gc = torch.compiler.config.force_cudagraph_gc
        torch.compiler.config.force_cudagraph_gc = True

    def tearDown(self):
        torch.compiler.config.force_cudagraph_gc = self._old_force_cudagraph_gc
        super().tearDown()


# ---------------------------------------------------------------------------
# Basic capture and replay -- no @no_graph functions.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestBasicCapture(unittest.TestCase):
    def test_basic_capture_and_replay(self):
        seq = CUDAGraphSequence()
        static_input = torch.empty(5, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_output = static_input * 2
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            static_output = static_input * 2

        for val in [3.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(static_output, torch.full((5,), val * 2, device="cuda"))
            )


# ---------------------------------------------------------------------------
# @no_graph placement -- where no-graph regions can occur in a capture.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestNoGraphPlacement(parameterized.TestCase):
    def test_no_graph_sandwich(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        def step1(dst: torch.Tensor, src: torch.Tensor):
            dst.copy_(src)

        @no_graph
        def step2(x: torch.Tensor):
            x.mul_(3.0)

        def step3(x: torch.Tensor):
            x.add_(1.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            step1(buf, src)
            step2(buf)
            step3(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        self.assertEqual(len(seq._segments), 3)
        self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[1], bcg._EagerSegment)
        self.assertIsInstance(seq._segments[2], torch.cuda.CUDAGraph)

        for val in [2.0, 4.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))
            )

    def test_no_graph_at_start(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        @no_graph
        def step1(dst: torch.Tensor, src: torch.Tensor):
            dst.copy_(src)

        def step2(x: torch.Tensor):
            x.mul_(3.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            step1(buf, src)
            step2(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        self.assertEqual(len(seq._segments), 3)
        self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[1], bcg._EagerSegment)
        self.assertIsInstance(seq._segments[2], torch.cuda.CUDAGraph)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0, device="cuda"))
            )

    def test_no_graph_at_end(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        def step1(dst: torch.Tensor, src: torch.Tensor):
            dst.copy_(src)
            dst.mul_(3.0)

        @no_graph
        def step2(x: torch.Tensor):
            x.add_(1.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            step1(buf, src)
            step2(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        self.assertEqual(len(seq._segments), 3)
        self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[1], bcg._EagerSegment)
        self.assertIsInstance(seq._segments[2], torch.cuda.CUDAGraph)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))
            )

    def test_many_no_graph_everywhere(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        def step1(dst: torch.Tensor, src: torch.Tensor):
            dst.copy_(src)

        @no_graph
        def step2(x: torch.Tensor):
            x.mul_(3.0)

        def step3(x: torch.Tensor):
            x.add_(1.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            force_no_graph()
            force_no_graph()
            step1(buf, src)
            force_no_graph()
            step2(buf)
            force_no_graph()
            step3(buf)
            force_no_graph()
            force_no_graph()

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        # 6 force_no_graph + 1 @no_graph = 7 no-graph regions
        # each adds 1 eager segment + 1 graph, starting with 1 graph: 1 + 7*2 = 15
        self.assertEqual(len(seq._segments), 15)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))
            )

    def test_force_no_graph(self):
        @no_graph
        def _user_defined_noop() -> None:
            pass

        for break_fn_name, break_fn in [
            ("builtin", force_no_graph),
            ("user_noop", _user_defined_noop),
        ]:
            with self.subTest(break_fn=break_fn_name):
                buf = torch.empty(5, device="cuda")

                def all_steps(x: torch.Tensor, break_fn=break_fn):
                    x.fill_(2.0)
                    break_fn()
                    x.mul_(3.0)
                    break_fn()
                    x.add_(1.0)

                seq = CUDAGraphSequence()

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        all_steps(buf)
                torch.cuda.current_stream().wait_stream(s)

                with breakable_graph(seq):
                    all_steps(buf)

                self.assertEqual(len(seq._segments), 5)
                self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
                self.assertIsInstance(seq._segments[1], bcg._EagerSegment)
                self.assertIsInstance(seq._segments[2], torch.cuda.CUDAGraph)
                self.assertIsInstance(seq._segments[3], bcg._EagerSegment)
                self.assertIsInstance(seq._segments[4], torch.cuda.CUDAGraph)

                seq.replay()
                self.assertTrue(torch.equal(buf, torch.full((5,), 7.0, device="cuda")))

    @parameterized.named_parameters(
        ("enabled", True, 3),
        ("disabled", False, 1),
    )
    def test_no_graph_enable_toggle(self, enable: bool, expected_segments: int):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        def step1(dst: torch.Tensor, src: torch.Tensor):
            dst.copy_(src)

        @no_graph(enable=enable)
        def step2(x: torch.Tensor):
            x.mul_(3.0)

        def step3(x: torch.Tensor):
            x.add_(1.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            step1(buf, src)
            step2(buf)
            step3(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        self.assertEqual(len(seq._segments), expected_segments)

        for val in [2.0, 4.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(
                    buf,
                    torch.full((5,), val * 3.0 + 1.0, device="cuda"),
                )
            )

    def test_no_graph_on_bound_method(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        class Scaler:
            def __init__(self, factor: float):
                self.factor = factor

            @no_graph
            def scale(self, x: torch.Tensor):
                x.mul_(self.factor)

        scaler = Scaler(3.0)

        def workload(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            scaler.scale(buf)
            buf.add_(1.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                workload(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            workload(buf, static_input)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0 + 1.0, device="cuda"))
            )


# ---------------------------------------------------------------------------
# Mixed operations -- gemm, element-wise, reductions with no-graph regions.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestMixedOps(ForceCUDAGraphGC, unittest.TestCase):
    def test_mixed_ops_multiple_no_graph(self):
        N = 64
        static_a = torch.empty(N, N, device="cuda")
        static_b = torch.empty(N, N, device="cuda")
        buf = torch.empty(N, N, device="cuda")
        scalar = torch.empty(1, device="cuda")

        def gemm():
            buf.copy_(static_a @ static_b)

        @no_graph
        def scale():
            buf.mul_(2.0)

        def bias():
            buf.add_(1.0)

        @no_graph
        def reduce():
            scalar.copy_(buf.sum().unsqueeze(0))

        def postprocess():
            buf.fill_(scalar[0])

        def all_steps():
            gemm()
            scale()
            bias()
            reduce()
            postprocess()

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_a.fill_(0.1)
                static_b.fill_(0.2)
                all_steps()
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps()

        # gemm | scale | bias + reduce | postprocess
        self.assertEqual(len(seq._segments), 5)
        self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[1], bcg._EagerSegment)
        self.assertIsInstance(seq._segments[2], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[3], bcg._EagerSegment)
        self.assertIsInstance(seq._segments[4], torch.cuda.CUDAGraph)

        for a_val, b_val in [(0.1, 0.2), (0.5, 0.3)]:
            static_a.fill_(a_val)
            static_b.fill_(b_val)
            seq.replay()
            expected_elem = a_val * b_val * N
            expected_elem = expected_elem * 2.0 + 1.0
            expected_sum = expected_elem * N * N
            self.assertTrue(
                torch.allclose(scalar, torch.tensor([expected_sum], device="cuda"))
            )

    def test_user_joined_stream_before_no_graph(self):
        """Side stream forked and joined by the user before a no-graph region.

        A properly joined side stream lets capture end cleanly at the no-graph
        boundary, so no error is raised. (Default mode does no fork tracking;
        this exercises the common, correctly-joined path.)
        """
        buf = torch.empty(5, device="cuda")
        side = torch.cuda.Stream()

        @no_graph
        def eager_step():
            buf.mul_(2.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(1.0)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    buf.add_(1.0)
                torch.cuda.current_stream().wait_stream(side)
                eager_step()
                buf.add_(10.0)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            buf.fill_(1.0)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                buf.add_(1.0)
            torch.cuda.current_stream().wait_stream(side)
            eager_step()
            buf.add_(10.0)

        seq.replay()
        self.assertTrue(torch.equal(buf, torch.full((5,), 14.0, device="cuda")))


# ---------------------------------------------------------------------------
# Return values -- @no_graph functions may return non-tensors; CUDA tensors raise.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestReturnValues(parameterized.TestCase):
    def test_no_graph_returns_tensor_raises(self):
        static_input = torch.empty(5, device="cuda")

        @no_graph
        def compute(x):
            return x * 3.0

        seq = CUDAGraphSequence()
        static_input.fill_(2.0)

        with self.assertRaisesRegex(RuntimeError, "returns one or more CUDA tensors"):
            with breakable_graph(seq):
                compute(static_input)

    @parameterized.named_parameters(
        ("list", lambda a, b: [a, b]),
        ("tuple", lambda a, b: (a, b)),
        ("dict", lambda a, b: {"x": a, "y": b}),
        ("dict_with_non_tensor", lambda a, b: {"x": a, "y": b, "s": 1.0}),
    )
    def test_no_graph_returns_structured_tensor_raises(self, pack):
        static_input = torch.empty(5, device="cuda")

        @no_graph
        def compute(x):
            return pack(x * 2.0, x * 3.0)

        seq = CUDAGraphSequence()
        static_input.fill_(2.0)

        with self.assertRaisesRegex(RuntimeError, "returns one or more CUDA tensors"):
            with breakable_graph(seq):
                compute(static_input)

    def test_no_graph_returns_non_tensor_allowed(self):
        static_input = torch.empty(5, device="cuda")

        @no_graph
        def compute(x):
            x.mul_(2.0)
            return 1.0

        seq = CUDAGraphSequence()
        static_input.fill_(2.0)

        with breakable_graph(seq):
            result = compute(static_input)

        self.assertEqual(result, 1.0)

    def test_no_graph_returns_cpu_tensor_allowed(self):
        static_input = torch.empty(5, device="cuda")

        @no_graph
        def compute(x):
            x.mul_(2.0)
            return torch.tensor(x.numel())

        seq = CUDAGraphSequence()
        static_input.fill_(2.0)

        with breakable_graph(seq):
            result = compute(static_input)

        self.assertFalse(result.is_cuda)
        self.assertEqual(result.item(), 5)

    def test_no_graph_with_args_kwargs(self):
        static_input = torch.empty(5, device="cuda")
        result_sum = torch.empty(5, device="cuda")

        @no_graph
        def compute(
            dst: torch.Tensor, *args: torch.Tensor, scale: float, extra: float
        ) -> float:
            dst.copy_(sum(args) * scale)
            return extra * 2.0

        def all_steps(scale: float, extra: float) -> float:
            a = static_input * 2.0
            b = static_input * 3.0
            extra = compute(result_sum, a, b, scale=scale, extra=extra)
            extra *= 3
            return extra

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(0.5, 0.1)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            extra = all_steps(0.5, 0.1)

        self.assertEqual(extra, 0.1 * 2.0 * 3)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(
                    result_sum,
                    torch.full((5,), (val * 2.0 + val * 3.0) * 0.5, device="cuda"),
                )
            )


# ---------------------------------------------------------------------------
# Edge cases -- outside capture, reset, nesting, exceptions.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestEdgeCases(unittest.TestCase):
    def test_no_graph_outside_capture(self):
        buf = torch.empty(5, device="cuda")

        @no_graph
        def step(x: torch.Tensor):
            x.mul_(3.0)

        buf.fill_(2.0)
        step(buf)
        self.assertTrue(torch.equal(buf, torch.full((5,), 6.0, device="cuda")))

        force_no_graph()

        buf.fill_(4.0)
        step(buf)
        self.assertTrue(torch.equal(buf, torch.full((5,), 12.0, device="cuda")))

    def test_reset_and_recapture(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        @no_graph
        def eager_mul(x: torch.Tensor):
            x.mul_(3.0)

        def workload_a(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            eager_mul(buf)
            buf.add_(1.0)

        def workload_b(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            eager_mul(buf)
            buf.mul_(2.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                workload_a(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            workload_a(buf, static_input)

        static_input.fill_(2.0)
        seq.replay()
        self.assertTrue(
            torch.equal(buf, torch.full((5,), 2.0 * 3.0 + 1.0, device="cuda"))
        )

        seq.reset()
        self.assertEqual(len(seq._segments), 0)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                workload_b(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            workload_b(buf, static_input)

        static_input.fill_(2.0)
        seq.replay()
        self.assertTrue(
            torch.equal(buf, torch.full((5,), 2.0 * 3.0 * 2.0, device="cuda"))
        )

    def test_no_graph_nested_call_stack(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        @no_graph
        def leaf_op(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.copy_(src * 3.0)

        def inner(dst: torch.Tensor, src: torch.Tensor) -> None:
            leaf_op(dst, src)
            dst.add_(1.0)

        def middle(dst: torch.Tensor, src: torch.Tensor) -> None:
            inner(dst, src)

        def outer(buf: torch.Tensor, src: torch.Tensor) -> None:
            middle(buf, src)
            buf.mul_(2.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                outer(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            outer(buf, static_input)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(
                    buf,
                    torch.full((5,), (val * 3.0 + 1.0) * 2.0, device="cuda"),
                )
            )

    def test_exception_during_capture(self):
        buf = torch.empty(5, device="cuda")

        @no_graph
        def failing_step(x: torch.Tensor):
            raise RuntimeError("intentional failure")

        def all_steps(buf: torch.Tensor):
            buf.fill_(2.0)
            failing_step(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(2.0)
        torch.cuda.current_stream().wait_stream(s)

        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with breakable_graph(seq):
                all_steps(buf)

        self.assertIsNone(bcg._current_breakable_graph_ctx.get())

    def test_exception_during_open_segment(self):
        buf = torch.empty(5, device="cuda")
        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(2.0)
        torch.cuda.current_stream().wait_stream(s)

        # Raise inside the body while a segment is still open (no @no_graph
        # break yet). __exit__ must let the original error propagate (not mask
        # it with a cleanup error) and still reset the context var.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with breakable_graph(seq):
                buf.fill_(2.0)
                raise RuntimeError("boom")

        self.assertIsNone(bcg._current_breakable_graph_ctx.get())

    def test_no_graph_with_cpu_tensor_arg(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")
        cpu_factor = torch.tensor(3.0)  # CPU tensor passed alongside CUDA args

        @no_graph
        def scale_by(dst: torch.Tensor, factor: torch.Tensor) -> None:
            # `factor` is non-CUDA: it must pass through pinning untouched
            # rather than hit the CUDA-only tensor constructor.
            dst.mul_(factor.item())

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.copy_(static_input)
                scale_by(buf, cpu_factor)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            buf.copy_(static_input)
            scale_by(buf, cpu_factor)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0, device="cuda"))
            )

    def test_no_graph_with_nested_container_args(self):
        a = torch.empty(5, device="cuda")
        b = torch.empty(5, device="cuda")
        weight = torch.empty(5, device="cuda")
        cpu_bias = torch.tensor(2.0)  # CPU tensor nested among CUDA args
        dst = torch.empty(5, device="cuda")

        @no_graph
        def combine(
            out: torch.Tensor,
            tensors: list[torch.Tensor],
            *,
            cfg: dict[str, torch.Tensor],
        ) -> None:
            # CUDA tensors nested in a list and a dict, plus a CPU tensor --
            # exercises tree_map pinning over containers and the CPU
            # passthrough in _make_replay_tensor_alias.
            out.copy_(
                (tensors[0] + tensors[1]) * cfg["weight"] + cfg["cpu_bias"].item()
            )

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                a.fill_(1.0)
                b.fill_(2.0)
                weight.fill_(3.0)
                combine(dst, [a, b], cfg={"weight": weight, "cpu_bias": cpu_bias})
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            combine(dst, [a, b], cfg={"weight": weight, "cpu_bias": cpu_bias})

        for a_val, b_val, w_val in [(1.0, 2.0, 3.0), (2.0, 5.0, 4.0)]:
            a.fill_(a_val)
            b.fill_(b_val)
            weight.fill_(w_val)
            seq.replay()
            expected = (a_val + b_val) * w_val + 2.0
            self.assertTrue(torch.equal(dst, torch.full((5,), expected, device="cuda")))

    def test_replay_after_partial_capture_failure(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        @no_graph
        def failing_step(x: torch.Tensor):
            raise RuntimeError("intentional failure")

        @no_graph
        def eager_step(x: torch.Tensor):
            x.mul_(3.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                buf.copy_(static_input)
        torch.cuda.current_stream().wait_stream(s)

        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with breakable_graph(seq):
                buf.copy_(static_input)
                failing_step(buf)

        seq.reset()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                buf.copy_(static_input)
                eager_step(buf)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            buf.copy_(static_input)
            eager_step(buf)

        for val in [2.0, 5.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), val * 3.0, device="cuda"))
            )

    def test_nested_breakable_graph_raises(self):
        buf = torch.empty(5, device="cuda")
        seq1 = CUDAGraphSequence()
        seq2 = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(2.0)
        torch.cuda.current_stream().wait_stream(s)

        with self.assertRaisesRegex(
            RuntimeError,
            "nested breakable_graph captures are not supported",
        ):
            with breakable_graph(seq1):
                buf.fill_(2.0)
                with breakable_graph(seq2):
                    buf.mul_(3.0)

    def test_exception_during_begin_segment(self):
        buf = torch.empty(5, device="cuda")

        call_count = 0

        @no_graph
        def eager_step(x: torch.Tensor):
            nonlocal call_count
            call_count += 1
            x.mul_(2.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(1.0)
                eager_step(buf)
        torch.cuda.current_stream().wait_stream(s)

        # Let the first segment (entered in __enter__) succeed, then fail when a
        # no-graph region tries to begin the next segment -- so __exit__ runs
        # with _graph_ctx already None.
        original_begin = bcg.breakable_graph._begin_segment
        begin_calls = 0

        def failing_begin(self):
            nonlocal begin_calls
            begin_calls += 1
            if begin_calls == 1:
                return original_begin(self)
            raise RuntimeError("simulated _begin_segment failure")

        with patch.object(bcg.breakable_graph, "_begin_segment", failing_begin):
            with self.assertRaisesRegex(
                RuntimeError, "simulated _begin_segment failure"
            ):
                with breakable_graph(seq):
                    buf.fill_(1.0)
                    eager_step(buf)

        self.assertIsNone(bcg._current_breakable_graph_ctx.get())


# ---------------------------------------------------------------------------
# is_in_breakable_graph -- query whether a capture is currently active.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestIsInBreakableGraph(unittest.TestCase):
    def test_is_in_breakable_graph(self):
        buf = torch.empty(5, device="cuda")
        observed = {}

        @no_graph
        def eager_step(x: torch.Tensor):
            # Runs in the eager window between two graph segments -- the capture
            # is still active, so this must report True during capture (and
            # False when this same body re-runs during replay).
            observed["eager_segment"] = is_in_breakable_graph()
            x.mul_(2.0)

        def all_steps(buf: torch.Tensor):
            buf.fill_(1.0)
            # Inside a captured graph segment.
            observed["graph_segment"] = is_in_breakable_graph()
            eager_step(buf)
            buf.add_(1.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                all_steps(buf)
        torch.cuda.current_stream().wait_stream(s)

        # Outside any capture.
        self.assertFalse(is_in_breakable_graph())

        with breakable_graph(seq):
            all_steps(buf)

        # Back outside the capture.
        self.assertFalse(is_in_breakable_graph())
        # True while capturing both a graph segment and an eager segment.
        self.assertTrue(observed["graph_segment"])
        self.assertTrue(observed["eager_segment"])

        # Replay does not enter breakable_graph, so it must report False.
        observed.clear()
        seq.replay()
        self.assertFalse(observed["eager_segment"])


# ---------------------------------------------------------------------------
# Memory pools -- sharing across sequences and within segments.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestMemoryPools(unittest.TestCase):
    def test_fresh_sequence_lazy_pool(self):
        seq = CUDAGraphSequence()
        self.assertEqual(seq._segments, [])
        handle = seq.pool()
        self.assertIsNotNone(handle)
        self.assertEqual(seq.pool(), handle)

    def test_shared_memory_pool(self):
        static_in_1 = torch.empty(5, device="cuda")
        static_in_2 = torch.empty(5, device="cuda")
        buf_1 = torch.empty(5, device="cuda")
        buf_2 = torch.empty(5, device="cuda")

        @no_graph
        def eager_scale(x: torch.Tensor):
            x.mul_(2.0)

        def workload_1(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            eager_scale(buf)
            buf.add_(1.0)

        def workload_2(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            eager_scale(buf)
            buf.add_(2.0)

        seq1 = CUDAGraphSequence()
        seq2 = CUDAGraphSequence(pool=seq1.pool())

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_in_1.fill_(10.0)
                static_in_2.fill_(20.0)
                workload_1(buf_1, static_in_1)
                workload_2(buf_2, static_in_2)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq1):
            workload_1(buf_1, static_in_1)

        with breakable_graph(seq2):
            workload_2(buf_2, static_in_2)

        self.assertEqual(seq1.pool(), seq2.pool())

        for v1, v2 in [(10.0, 20.0), (3.0, 7.0)]:
            static_in_1.fill_(v1)
            static_in_2.fill_(v2)
            seq1.replay()
            seq2.replay()
            self.assertTrue(
                torch.equal(
                    buf_1,
                    torch.full((5,), v1 * 2.0 + 1.0, device="cuda"),
                )
            )
            self.assertTrue(
                torch.equal(
                    buf_2,
                    torch.full((5,), v2 * 2.0 + 2.0, device="cuda"),
                )
            )

    def test_intra_sequence_pool_sharing(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        @no_graph
        def eager_step(x: torch.Tensor):
            x.mul_(3.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            eager_step(buf)
            buf.add_(1.0)
            eager_step(buf)
            buf.add_(2.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        graphs = [s for s in seq._segments if isinstance(s, torch.cuda.CUDAGraph)]
        self.assertEqual(len(graphs), 3)
        self.assertEqual(graphs[0].pool(), graphs[1].pool())
        self.assertEqual(graphs[1].pool(), graphs[2].pool())

    def test_pool_persists_after_reset(self):
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")

        @no_graph
        def eager_step(x: torch.Tensor):
            x.mul_(3.0)

        def workload(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            eager_step(buf)
            buf.add_(1.0)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                workload(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            workload(buf, static_input)

        pool_before = seq.pool()
        seq.reset()

        self.assertEqual(seq.pool(), pool_before)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(2.0)
                workload(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            workload(buf, static_input)

        self.assertEqual(seq.pool(), pool_before)

        static_input.fill_(4.0)
        seq.replay()
        self.assertTrue(
            torch.equal(buf, torch.full((5,), 4.0 * 3.0 + 1.0, device="cuda"))
        )


# ---------------------------------------------------------------------------
# Drop-in replacement -- breakable_graph used in place of torch.cuda.graph.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestDropInReplacement(unittest.TestCase):
    def test_whole_network_capture_drop_in(self):
        N, D_in, H, D_out = 640, 4096, 2048, 1024
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(H, D_out),
            torch.nn.Dropout(p=0.1),
        ).cuda()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        static_input = torch.randn(N, D_in, device="cuda")
        static_target = torch.randn(N, D_out, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(static_input)
                loss = loss_fn(y_pred, static_target)
                loss.backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(s)

        seq = CUDAGraphSequence()
        optimizer.zero_grad(set_to_none=True)
        with breakable_graph(seq):
            static_y_pred = model(static_input)
            static_loss = loss_fn(static_y_pred, static_target)
            static_loss.backward()
            optimizer.step()

        real_inputs = [torch.rand_like(static_input) for _ in range(10)]
        real_targets = [torch.rand_like(static_target) for _ in range(10)]

        for data, target in zip(real_inputs, real_targets):
            static_input.copy_(data)
            static_target.copy_(target)
            seq.replay()

        self.assertEqual(static_y_pred.shape, (N, D_out))
        self.assertEqual(static_loss.ndim, 0)

    def test_amp_with_graph_capture_drop_in(self):
        N, D_in, H, D_out = 640, 4096, 2048, 1024
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(H, D_out),
            torch.nn.Dropout(p=0.1),
        ).cuda()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.GradScaler()

        static_input = torch.randn(N, D_in, device="cuda")
        static_target = torch.randn(N, D_out, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    y_pred = model(static_input)
                    loss = loss_fn(y_pred, static_target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        torch.cuda.current_stream().wait_stream(s)

        seq = CUDAGraphSequence()
        optimizer.zero_grad(set_to_none=True)
        with breakable_graph(seq):
            with torch.amp.autocast("cuda"):
                static_y_pred = model(static_input)
                static_loss = loss_fn(static_y_pred, static_target)
            scaler.scale(static_loss).backward()

        real_inputs = [torch.rand_like(static_input) for _ in range(10)]
        real_targets = [torch.rand_like(static_target) for _ in range(10)]

        for data, target in zip(real_inputs, real_targets):
            static_input.copy_(data)
            static_target.copy_(target)
            seq.replay()
            scaler.step(optimizer)
            scaler.update()

        self.assertEqual(static_y_pred.shape, (N, D_out))
        self.assertEqual(static_loss.ndim, 0)

    def test_eager_break_between_forward_and_backward(self):
        """Showcase autograd across an @no_graph break.

        Captures forward and backward into separate CUDA graph segments split by
        an eager @no_graph region that reads the loss to host (a common reason to
        drop to eager mid-step, e.g. logging). Verifies the replayed gradients
        match an eager reference, and that the eager break observes the live,
        per-replay loss value.
        """
        torch.manual_seed(0)
        N, D_in, H, D_out = 64, 256, 128, 32
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        ).cuda()
        loss_fn = torch.nn.MSELoss()

        static_input = torch.randn(N, D_in, device="cuda")
        static_target = torch.randn(N, D_out, device="cuda")

        loss_log: list[float] = []

        @no_graph
        def log_loss(loss: torch.Tensor) -> None:
            # Eager break between forward and backward. Reading the scalar loss
            # to host forces a device sync, so it must run outside the graph.
            # Returning a CUDA tensor here would raise; a host float is fine.
            loss_log.append(loss.item())

        # Warm up on a side stream (required before capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                loss = loss_fn(model(static_input), static_target)
                log_loss(loss)
                loss.backward()
        torch.cuda.current_stream().wait_stream(s)

        seq = CUDAGraphSequence()
        model.zero_grad(set_to_none=True)
        with breakable_graph(seq):
            static_y = model(static_input)
            static_loss = loss_fn(static_y, static_target)
            log_loss(static_loss)  # eager break splits fwd graph from bwd graph
            static_loss.backward()

        # forward graph -> eager break -> backward graph
        self.assertEqual(len(seq._segments), 3)
        self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[1], bcg._EagerSegment)
        self.assertIsInstance(seq._segments[2], torch.cuda.CUDAGraph)

        loss_log.clear()
        for _ in range(3):
            data = torch.randn_like(static_input)
            target = torch.randn_like(static_target)
            static_input.copy_(data)
            static_target.copy_(target)

            # Static grad buffers accumulate across replays; zero them in place
            # (set_to_none would detach the graph's output buffers).
            for p in model.parameters():
                p.grad.zero_()
            seq.replay()
            torch.cuda.synchronize()

            # Eager reference on the same weights and data.
            ref_loss = loss_fn(model(data), target)
            ref_grads = torch.autograd.grad(ref_loss, list(model.parameters()))

            for p, ref in zip(model.parameters(), ref_grads):
                torch.testing.assert_close(p.grad, ref, rtol=1e-3, atol=1e-3)

            # The eager break saw the live loss from this replay's forward.
            torch.testing.assert_close(
                loss_log[-1], ref_loss.item(), rtol=1e-3, atol=1e-3
            )

        self.assertEqual(len(loss_log), 3)

    def test_whole_training_step_with_eager_layer_fwd_and_bwd(self):
        """Showcase a full training step under breakable_graph where one model
        layer runs eagerly in BOTH its forward and backward.

        The eager layer is a custom autograd Function whose forward and backward
        each call an @no_graph helper, so capture splits the step into:
        forward graph -> eager forward -> backward graph -> eager backward ->
        backward graph + optimizer step. Replaying the captured sequence is
        verified to match eager SGD steps parameter-for-parameter.
        """
        torch.manual_seed(0)
        N, D_in, H, D_out = 64, 256, 128, 32

        @no_graph
        def eager_relu_fwd(out: torch.Tensor, x: torch.Tensor) -> None:
            # Runs eagerly between graph segments; writes into a static buffer
            # instead of returning a CUDA tensor (which @no_graph forbids).
            out.copy_(torch.relu(x))

        @no_graph
        def eager_relu_bwd(
            grad_in: torch.Tensor, grad_out: torch.Tensor, x: torch.Tensor
        ) -> None:
            grad_in.copy_(grad_out * (x > 0))

        class EagerReLU(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, fwd_out, bwd_out):
                ctx.save_for_backward(x)
                ctx.bwd_out = bwd_out
                eager_relu_fwd(fwd_out, x)  # eager break during forward
                return fwd_out

            @staticmethod
            def backward(ctx, grad_out):
                (x,) = ctx.saved_tensors
                eager_relu_bwd(ctx.bwd_out, grad_out, x)  # eager break in backward
                return ctx.bwd_out, None, None

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(D_in, H)
                self.l2 = torch.nn.Linear(H, D_out)
                # Static buffers the eager layer writes into across replays.
                self.register_buffer("fwd_out", torch.empty(N, H))
                self.register_buffer("bwd_out", torch.empty(N, H))

            def forward(self, x):
                h = self.l1(x)
                h = EagerReLU.apply(h, self.fwd_out, self.bwd_out)
                return self.l2(h)

        model = Net().cuda()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        static_input = torch.randn(N, D_in, device="cuda")
        static_target = torch.randn(N, D_out, device="cuda")

        # Warm up on a side stream (required before capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(static_input), static_target)
                loss.backward()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(s)

        # Capture the whole training step: forward + backward + optimizer step.
        # Backward must run single-threaded: the autograd engine otherwise
        # dispatches backward nodes to a worker thread where the breakable_graph
        # context (a contextvar) is not visible, so the @no_graph break in the
        # custom backward would not fire. Running on the capturing thread keeps
        # both the context and the stream-capture state consistent.
        seq = CUDAGraphSequence()
        optimizer.zero_grad(set_to_none=True)
        with breakable_graph(seq):
            static_pred = model(static_input)
            static_loss = loss_fn(static_pred, static_target)
            with torch.autograd.set_multithreading_enabled(False):
                static_loss.backward()
            optimizer.step()

        # Two eager breaks -- one in the forward, one in the backward -- both
        # interior, with captured graph segments on either side.
        eager_idxs = [
            i
            for i, seg in enumerate(seq._segments)
            if isinstance(seg, bcg._EagerSegment)
        ]
        self.assertEqual(len(eager_idxs), 2)
        self.assertGreater(eager_idxs[0], 0)
        self.assertLess(eager_idxs[-1], len(seq._segments) - 1)
        self.assertIsInstance(seq._segments[0], torch.cuda.CUDAGraph)
        self.assertIsInstance(seq._segments[-1], torch.cuda.CUDAGraph)

        # Reference model that takes the same SGD steps eagerly.
        ref = copy.deepcopy(model)
        ref_optimizer = torch.optim.SGD(ref.parameters(), lr=0.1)

        for _ in range(3):
            data = torch.randn_like(static_input)
            target = torch.randn_like(static_target)
            static_input.copy_(data)
            static_target.copy_(target)

            # Static grads accumulate across replays; zero in place before step.
            for p in model.parameters():
                p.grad.zero_()
            seq.replay()

            ref_optimizer.zero_grad(set_to_none=True)
            ref_loss = loss_fn(ref(data), target)
            ref_loss.backward()
            ref_optimizer.step()
            torch.cuda.synchronize()

            for p, p_ref in zip(model.parameters(), ref.parameters()):
                torch.testing.assert_close(p, p_ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Debug-mode fork tracking -- unjoined side streams are named in the error.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestDebugForkTracking(ForceCUDAGraphGC, unittest.TestCase):
    def test_partial_join_reports_only_unjoined_stream(self):
        """Debug mode names the unjoined stream and omits the joined one.

        Forks two side streams, joins only one, then enters a no-graph region.
        The error must report the unjoined stream's id and not the joined one --
        verifying fork/join tracking discards joined streams.
        """
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")
        side1 = torch.cuda.Stream()
        side2 = torch.cuda.Stream()

        @no_graph
        def eager_step(x: torch.Tensor):
            x.mul_(2.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            side1.wait_stream(torch.cuda.current_stream())  # fork side1
            side2.wait_stream(torch.cuda.current_stream())  # fork side2
            with torch.cuda.stream(side1):
                buf.add_(1.0)
            with torch.cuda.stream(side2):
                buf.add_(2.0)
            torch.cuda.current_stream().wait_stream(side1)  # join side1 only
            eager_step(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(1.0)
                all_steps(buf, static_input)
                torch.cuda.current_stream().wait_stream(side2)
        torch.cuda.current_stream().wait_stream(s)

        with patch.object(bcg, "_DEBUG", True):
            with self.assertRaises(RuntimeError) as cm:
                with breakable_graph(seq):
                    all_steps(buf, static_input)

        msg = str(cm.exception)
        self.assertIn("was not joined back to the capturing stream", msg)
        self.assertIn("Unjoined side-stream id(s)", msg)
        self.assertIn(str(side2.cuda_stream), msg)
        self.assertNotIn(str(side1.cuda_stream), msg)

    def test_fork_join_fork_again_reports_unjoined_stream(self):
        """Debug mode tracks a re-fork: fork, join, then fork again without join.

        The discard-then-re-add tracking must still flag the final unjoined fork
        before the no-graph region.
        """
        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")
        side = torch.cuda.Stream()

        @no_graph
        def eager_step(x: torch.Tensor):
            x.mul_(2.0)

        def all_steps(buf: torch.Tensor, src: torch.Tensor):
            buf.copy_(src)
            side.wait_stream(torch.cuda.current_stream())  # fork
            with torch.cuda.stream(side):
                buf.add_(1.0)
            torch.cuda.current_stream().wait_stream(side)  # join
            side.wait_stream(torch.cuda.current_stream())  # fork again
            with torch.cuda.stream(side):
                buf.add_(2.0)
            eager_step(buf)

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(1.0)
                all_steps(buf, static_input)
                torch.cuda.current_stream().wait_stream(side)
        torch.cuda.current_stream().wait_stream(s)

        with patch.object(bcg, "_DEBUG", True):
            with self.assertRaisesRegex(RuntimeError, "Unjoined side-stream id"):
                with breakable_graph(seq):
                    all_steps(buf, static_input)


# ---------------------------------------------------------------------------
# Fork/join -- a side stream forked and joined within a capture, across the
# wait_stream / event / mixed join APIs.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestForkJoin(ForceCUDAGraphGC, parameterized.TestCase):
    @parameterized.named_parameters(
        ("wait_stream", "wait_stream", "wait_stream"),
        ("events", "event", "event"),
        ("mixed", "event", "wait_stream"),
    )
    def test_fork_join_variants(self, fork_method: str, join_method: str):
        side = torch.cuda.Stream()

        def fork_wait_stream():
            side.wait_stream(torch.cuda.current_stream())

        def join_wait_stream():
            torch.cuda.current_stream().wait_stream(side)

        def fork_event():
            e = torch.cuda.Event()
            e.record(torch.cuda.current_stream())
            e.wait(side)

        def join_event():
            e = torch.cuda.Event()
            e.record(side)
            e.wait(torch.cuda.current_stream())

        forks = {
            "event": fork_event,
            "wait_stream": fork_wait_stream,
        }
        joins = {
            "event": join_event,
            "wait_stream": join_wait_stream,
        }
        fork = forks[fork_method]
        join = joins[join_method]

        def all_steps(buf, src):
            buf.copy_(src)
            fork()
            with torch.cuda.stream(side):
                buf.add_(1.0)
            join()
            buf.mul_(2.0)

        static_input = torch.empty(5, device="cuda")
        buf = torch.empty(5, device="cuda")
        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                static_input.fill_(3.0)
                all_steps(buf, static_input)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            all_steps(buf, static_input)

        for val in [3.0, 7.0]:
            static_input.fill_(val)
            seq.replay()
            self.assertTrue(
                torch.equal(buf, torch.full((5,), (val + 1.0) * 2.0, device="cuda"))
            )


# ---------------------------------------------------------------------------
# Unjoined stream error messages -- clear errors when a side stream is left
# unjoined (default mode, no fork tracking).
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestUnjoinedStreamErrors(ForceCUDAGraphGC, parameterized.TestCase):
    @parameterized.named_parameters(
        ("wait_stream", "wait_stream"),
        ("event", "event"),
    )
    def test_unjoined_stream_error_at_no_graph(self, fork_method: str):
        """A side stream left unjoined at a no-graph boundary raises -- whether
        it was forked via wait_stream or via an event."""
        side = torch.cuda.Stream()

        def fork_wait_stream():
            side.wait_stream(torch.cuda.current_stream())

        def fork_event():
            e = torch.cuda.Event()
            e.record(torch.cuda.current_stream())
            e.wait(side)

        forks = {
            "event": fork_event,
            "wait_stream": fork_wait_stream,
        }
        fork = forks[fork_method]

        @no_graph
        def eager_step(b: torch.Tensor):
            b.mul_(2.0)

        buf = torch.empty(5, device="cuda")
        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(1.0)
                fork()
                with torch.cuda.stream(side):
                    buf.add_(1.0)
                torch.cuda.current_stream().wait_stream(side)
                eager_step(buf)
        torch.cuda.current_stream().wait_stream(s)

        with self.assertRaisesRegex(
            RuntimeError, "was not joined back to the capturing stream"
        ):
            with breakable_graph(seq):
                buf.fill_(1.0)
                fork()
                with torch.cuda.stream(side):
                    buf.add_(1.0)
                eager_step(buf)

    def test_unjoined_stream_error_at_end_of_capture(self):
        buf = torch.empty(5, device="cuda")
        side = torch.cuda.Stream()

        seq = CUDAGraphSequence()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                buf.fill_(1.0)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    buf.add_(1.0)
                torch.cuda.current_stream().wait_stream(side)
        torch.cuda.current_stream().wait_stream(s)

        with self.assertRaisesRegex(
            RuntimeError, "was not joined back to the capturing stream"
        ):
            with breakable_graph(seq):
                buf.fill_(1.0)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    buf.add_(1.0)


# ---------------------------------------------------------------------------
# Concurrent captures -- two threads capturing with no-graph regions
# simultaneously.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestConcurrentCaptures(unittest.TestCase):
    def test_concurrent_captures_with_no_graphs(self):
        barrier = threading.Barrier(2)
        results = {}
        errors = {}

        def capture_and_replay(name: str, multiplier: float, stream: torch.cuda.Stream):
            try:
                static_input = torch.empty(5, device="cuda")
                buf = torch.empty(5, device="cuda")

                @no_graph
                def eager_step(x: torch.Tensor, factor: float):
                    x.mul_(factor)

                def all_steps(buf: torch.Tensor, src: torch.Tensor):
                    buf.copy_(src)
                    eager_step(buf, multiplier)

                seq = CUDAGraphSequence()

                with torch.cuda.stream(stream):
                    for _ in range(3):
                        static_input.fill_(2.0)
                        all_steps(buf, static_input)
                torch.cuda.synchronize()

                barrier.wait()

                with breakable_graph(
                    seq,
                    stream=stream,
                    capture_error_mode="thread_local",
                ):
                    all_steps(buf, static_input)

                static_input.fill_(4.0)
                seq.replay()
                torch.cuda.synchronize()
                results[name] = buf.clone()
            except Exception as e:
                errors[name] = e

        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        t1 = threading.Thread(target=capture_and_replay, args=("t1", 3.0, s1))
        t2 = threading.Thread(target=capture_and_replay, args=("t2", 5.0, s2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertFalse(errors, f"Thread errors: {errors}")
        self.assertTrue(
            torch.equal(results["t1"], torch.full((5,), 12.0, device="cuda"))
        )
        self.assertTrue(
            torch.equal(results["t2"], torch.full((5,), 20.0, device="cuda"))
        )


# ---------------------------------------------------------------------------
# Memory footprint -- eager segments must retain only non-owning tensor pins.
# ---------------------------------------------------------------------------


@pytest.mark.gpus_needed_1
@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestEagerSegmentRetention(ForceCUDAGraphGC, unittest.TestCase):
    def test_eager_segment_does_not_retain_input_tensors(self):
        @no_graph
        def consume(x: torch.Tensor):
            x.add_(1.0)

        seq = CUDAGraphSequence()
        victim = torch.empty(5, device="cuda")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                consume(victim)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            consume(victim)

        ref = weakref.ref(victim)
        del victim
        gc.collect()
        self.assertIsNone(
            ref(),
            "eager segment retained the original input tensor; its buffer can "
            "no longer be reused by the allocator",
        )

    def test_pin_does_not_block_memory_reuse(self):
        @no_graph
        def consume(x: torch.Tensor):
            x.mul_(2.0)

        numel = 4099  # odd size, unlikely to collide with other alloc blocks
        seq = CUDAGraphSequence()
        buf = torch.empty(numel, device="cuda")
        ptr = buf.data_ptr()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                consume(buf)
        torch.cuda.current_stream().wait_stream(s)

        with breakable_graph(seq):
            consume(buf)

        del buf
        gc.collect()

        reused = torch.empty(numel, device="cuda")
        self.assertEqual(reused.data_ptr(), ptr)


class TestEndSegmentErrorAttribution(unittest.TestCase):
    """`_end_segment` should blame an unjoined side stream only when the CUDA
    error says so; other capture failures must surface unchanged. These run on
    CPU -- they drive `_end_segment` with a fake graph context.
    """

    def _breakable_graph_with_failing_ctx(self, exc: Exception) -> breakable_graph:
        pg = breakable_graph(CUDAGraphSequence())

        class _FakeCtx:
            def __exit__(self, *args):
                raise exc

        pg._graph_ctx = _FakeCtx()
        return pg

    def test_unjoined_error_is_reattributed(self):
        pg = self._breakable_graph_with_failing_ctx(
            RuntimeError("CUDA error: ... cudaErrorStreamCaptureUnjoined ...")
        )
        with self.assertRaisesRegex(
            RuntimeError, "was not joined back to the capturing stream"
        ):
            pg._end_segment()

    def test_other_capture_error_propagates(self):
        pg = self._breakable_graph_with_failing_ctx(
            RuntimeError("some unrelated capture failure")
        )
        with self.assertRaisesRegex(RuntimeError, "some unrelated capture failure"):
            pg._end_segment()
