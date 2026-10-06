"""HardSigmoidSequence: the fixed piecewise HardSigmoid sequence unit.

Mirrors the TanhSequence/HardSwishSequence conventions: independent
reference formulas, finite-difference gradient checks (including
trajectories that cross all three activation branches), stream/batch
equivalence, checkpoint and state-migration rules with a kind tag distinct
from every other sequence class, validation and failure atomicity, plus
HardSigmoid-specific numerics (exact 0.0 at and below z == -2.5, exact 1.0
at and above z == 2.5, fixed endpoint derivatives, per-step pre-activation
recording because the cached output alone cannot decide the branch: a z
just inside the open interval can round to exactly 0.0 or 1.0).
"""
import copy
import json
import math
import unittest

import app
from app import (ELUSequence, GELUSequence, HardSigmoidSequence,
                 HardSwishSequence, LeakyReLUSequence, Linear, MishSequence,
                 ReLUSequence, SELUSequence, SiLUSequence, SigmoidSequence,
                 SoftplusSequence, SoftsignSequence, TanhSequence)

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400  # finite int, but overflows double arithmetic

EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-7

W, B = [0.4, -0.3], 0.15
ROWS = [[0.8], [-0.5], [1.2], [-0.7], [0.3]]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]

# Configuration whose trajectory visits all three branches.
WB, BB = [2.0, 0.8], 0.5
BROWS = [[2.0], [-3.0], [0.2], [-2.5], [3.0]]

OTHER_CLASSES = (TanhSequence, SigmoidSequence, SoftplusSequence,
                 ReLUSequence, LeakyReLUSequence, ELUSequence, GELUSequence,
                 SELUSequence, SiLUSequence, MishSequence, SoftsignSequence,
                 HardSwishSequence)


def close(a, b, atol=ATOL, rtol=RTOL):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def allclose(xs, ys, atol=ATOL, rtol=RTOL):
    return len(xs) == len(ys) and all(
        close(a, b, atol, rtol) for a, b in zip(xs, ys)
    )


def central(f, eps=EPS):
    return (f(eps) - f(-eps)) / (2.0 * eps)


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("ValueError not raised by %r" % fn)


# ---------------------------------------------------------------------------
# Independent reference formulas, re-derived from the published recurrence.
# ---------------------------------------------------------------------------

def hardsigmoid(z):
    if z <= -2.5:
        return 0.0
    if z >= 2.5:
        return 1.0
    return z / 5 + 0.5


def hardsigmoid_derivative(z):
    if z <= -2.5:
        return 0.0
    if z >= 2.5:
        return 0.0
    return 0.2


def sequence_outputs(weight, bias, rows, h0, truncate, carry, starts=(),
                     frozen=None):
    """hardsigmoid(w_in.row + w_h*h_prev + b), with the published boundary
    rules. ``starts`` holds explicit segment-start indices; ``frozen`` maps
    a segment-start index to the detached constant hidden value consumed
    there (an explicit seed takes precedence, including at index 0)."""
    outs = []
    hidden = h0
    for i, row in enumerate(rows):
        if (truncate is not None and i % truncate == 0) or i in starts:
            # An explicit seed (including at position 0) detaches the
            # standard rule; otherwise step 0 consumes h0 directly so a
            # finite-difference perturbation of h0 is visible.
            if i == 0:
                hidden = frozen[0] if frozen is not None and 0 in frozen else h0
            elif frozen is not None and i in frozen:
                hidden = frozen[i]
            elif carry:
                hidden = outs[-1]
            else:
                hidden = 0.0
        z = sum(w * a for w, a in zip(weight[:-1], row)) \
            + weight[-1] * hidden + bias
        hidden = hardsigmoid(z)
        outs.append(hidden)
    return outs


def boundary_indices(n, truncate, starts=()):
    return {i for i in range(n)
            if (truncate is not None and i % truncate == 0) or i in starts}


def frozen_boundaries(weight, bias, rows, truncate, carry, h0,
                      starts=(), seeds=None):
    """Detached constants for the boundaries past index 0, plus any explicit
    seed at index 0. The plain h0 at index 0 is deliberately not frozen so
    the initial-hidden finite difference stays visible. Carry constants are
    taken from the trajectory the explicit seeds themselves produce."""
    explicit = {}
    if seeds is not None:
        explicit = {i: seed for i, seed in enumerate(seeds)
                    if seed is not None}
    base = sequence_outputs(weight, bias, rows, h0, truncate, carry, starts,
                            frozen=explicit if explicit else None)
    frozen = dict(explicit)
    for i in boundary_indices(len(rows), truncate, starts):
        if i > 0 and i not in frozen:
            frozen[i] = base[i - 1] if carry else 0.0
    return frozen


def objective(weight, bias, rows, h0, grad_outputs, grad_hidden,
              truncate, carry, starts=(), frozen=None):
    """L = sum_t go_t * out_t + grad_hidden * out_{-1} (grad_hidden * h0
    when the sequence is empty)."""
    outs = sequence_outputs(weight, bias, rows, h0, truncate, carry,
                            starts, frozen)
    if not outs:
        return grad_hidden * h0
    return sum(g * o for g, o in zip(grad_outputs, outs)) \
        + grad_hidden * outs[-1]


def fresh(weight=W, bias=B):
    return HardSigmoidSequence(Linear(list(weight), bias))


class SmokeTest(unittest.TestCase):
    def test_module_exposes_hardsigmoid_sequence(self):
        self.assertTrue(hasattr(app, "HardSigmoidSequence"))
        self.assertTrue(issubclass(HardSigmoidSequence, TanhSequence))

    def test_construction_rules_match_tanh(self):
        expect_value_error(lambda: HardSigmoidSequence("not a linear"))
        expect_value_error(lambda: HardSigmoidSequence(object()))
        one_weight = Linear([1.0])
        before = (list(one_weight.weight), one_weight.bias,
                  list(one_weight.grad), one_weight.grad_bias,
                  one_weight.last)
        expect_value_error(lambda: HardSigmoidSequence(one_weight))
        # A rejected constructor leaves the passed Linear exactly as it was.
        self.assertEqual((list(one_weight.weight), one_weight.bias,
                          list(one_weight.grad), one_weight.grad_bias,
                          one_weight.last), before)
        seq = HardSigmoidSequence(Linear([0.4, -0.3], 0.15))
        self.assertEqual(seq.d, 1)
        self.assertEqual(HardSigmoidSequence(Linear([0.4, -0.3, 0.2])).d, 2)
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])


class ActivationTest(unittest.TestCase):
    def test_activation_matches_reference_formula(self):
        for z in (-5.0, -2.5, -2.4, -1.0, -0.1, 0.0, 0.1, 2.4, 2.5, 5.0):
            self.assertTrue(close(HardSigmoidSequence._activate(z),
                                  hardsigmoid(z), atol=0.0),
                            "z=%r" % z)

    def test_fixed_branches_and_endpoint_values(self):
        # z <= -2.5: exact 0.0; -2.5 < z < 2.5: z/5 + 0.5; z >= 2.5: exact 1.0.
        for z in (-2.5, -3.0, -100.0, -1e300):
            self.assertEqual(HardSigmoidSequence._activate(z), 0.0)
        for z in (2.5, 3.0, 100.0, 1e300):
            self.assertEqual(HardSigmoidSequence._activate(z), 1.0)
        for z in (-2.499999, -1.0, 0.0, 2.499999):
            self.assertEqual(HardSigmoidSequence._activate(z),
                             z / 5 + 0.5)
        # The endpoints take the saturated branches, not the middle one.
        self.assertEqual(HardSigmoidSequence._activate(-2.5), 0.0)
        self.assertEqual(HardSigmoidSequence._activate(2.5), 1.0)
        self.assertEqual(HardSigmoidSequence._activate(0.0), 0.5)

    def test_extreme_preactivations_stay_finite(self):
        for z in (1e150, -1e150, 1e300, -1e300, 1.7e308, -1.7e308):
            out = HardSigmoidSequence._activate(z)
            self.assertTrue(math.isfinite(out), "z=%r -> %r" % (z, out))
        self.assertEqual(HardSigmoidSequence._activate(1e300), 1.0)
        self.assertEqual(HardSigmoidSequence._activate(-1e300), 0.0)

    def test_derivative_matches_formula_and_finite_difference(self):
        # Exact branch values, including the fixed endpoints.
        for z in (-10.0, -2.5, 2.5, 10.0):
            self.assertEqual(HardSigmoidSequence._activation_derivative(z), 0.0)
        # Interior points away from the kinks all carry the fixed slope 0.2.
        for z in (-2.4, -2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, 2.4):
            got = HardSigmoidSequence._activation_derivative(z)
            self.assertEqual(got, 0.2)
            fd = central(lambda dlt, z=z: hardsigmoid(z + dlt), eps=1e-6)
            self.assertTrue(close(got, fd, atol=1e-6, rtol=1e-6),
                            "z=%r: %r vs fd %r" % (z, got, fd))

    def test_derivative_stays_finite_for_extreme_z(self):
        for z in (1e150, -1e150, 1e300, -1e300, 1.7e308, -1.7e308):
            got = HardSigmoidSequence._activation_derivative(z)
            self.assertTrue(math.isfinite(got), "z=%r -> %r" % (z, got))
            self.assertEqual(got, 0.0)


class ForwardTest(unittest.TestCase):
    def test_forward_matches_reference_recurrence(self):
        for truncate, carry, h0 in ((None, False, 0.0), (2, False, 0.0),
                                    (2, True, 0.2), (3, False, -0.4),
                                    (1, True, 0.0)):
            seq = fresh()
            got = seq.forward(ROWS, truncate=truncate, carry_hidden=carry,
                              initial_hidden=h0)
            want = sequence_outputs(W, B, ROWS, h0, truncate, carry)
            self.assertTrue(allclose(got, want, atol=1e-15),
                            "%r vs %r" % (got, want))
            self.assertTrue(close(seq.hidden, want[-1], atol=1e-15))
            self.assertEqual(seq.outputs, got)

    def test_forward_crosses_all_activation_branches(self):
        for truncate, carry in ((None, False), (2, False), (2, True)):
            seq = fresh(WB, BB)
            got = seq.forward(BROWS, truncate=truncate, carry_hidden=carry)
            want = sequence_outputs(WB, BB, BROWS, 0.0, truncate, carry)
            self.assertEqual(got, want)
            # The untruncated trajectory provably visits every branch.
            if truncate is None:
                zs = []
                hidden = 0.0
                for row in BROWS:
                    z = WB[0] * row[0] + WB[1] * hidden + BB
                    zs.append(z)
                    hidden = hardsigmoid(z)
                self.assertTrue(any(z <= -2.5 for z in zs))
                self.assertTrue(any(-2.5 < z < 2.5 for z in zs))
                self.assertTrue(any(z >= 2.5 for z in zs))

    def test_forward_with_segment_starts_merges_boundaries(self):
        starts = [False, True, False, False, True]
        for carry in (False, True):
            seq = fresh()
            got = seq.forward(ROWS, truncate=3, carry_hidden=carry,
                              initial_hidden=0.2, segment_starts=starts)
            want = sequence_outputs(W, B, ROWS, 0.2, 3, carry,
                                    starts={1, 4})
            self.assertTrue(allclose(got, want, atol=1e-15))

    def test_forward_with_segment_hiddens(self):
        # Explicit detached seeds at the truncate boundaries 0 and 2.
        seeds = [0.35, None, -0.25, None, None]
        seq = fresh()
        got = seq.forward(ROWS, truncate=2, carry_hidden=True,
                          segment_hiddens=seeds)
        want = []
        hidden = 0.0
        for i, row in enumerate(ROWS):
            if i % 2 == 0:
                hidden = seeds[i] if seeds[i] is not None else (
                    want[-1] if i > 0 else 0.0)
            z = W[0] * row[0] + W[1] * hidden + B
            hidden = hardsigmoid(z)
            want.append(hidden)
        self.assertEqual(got, want)
        # A seed away from a boundary is rejected before any state changes.
        bad = fresh()
        bad.forward(ROWS, truncate=2)
        snap = bad.outputs
        expect_value_error(lambda: bad.forward(
            ROWS, truncate=2, segment_hiddens=[None, 0.5, None, None, None]))
        self.assertEqual(bad.outputs, snap)

    def test_multi_feature_forward_matches_reference(self):
        weight, bias = [0.4, -0.3, 0.2], 0.1
        rows = [[0.5, 0.1], [-0.2, 0.7], [0.9, -0.4]]
        seq = fresh(weight, bias)
        got = seq.forward(rows, truncate=2)
        want = sequence_outputs(weight, bias, rows, 0.0, 2, False)
        self.assertTrue(allclose(got, want, atol=1e-15))

    def test_empty_forward_commits_initial_hidden(self):
        seq = fresh()
        self.assertEqual(seq.forward([], initial_hidden=0.3), [])
        self.assertEqual(seq.hidden, 0.3)
        self.assertEqual(seq.outputs, [])


class BackwardTest(unittest.TestCase):
    def _check_pass(self, weight, bias, rows, h0, truncate, carry,
                    grad_hidden, starts=None, seeds=None):
        go = [0.3 * (-1) ** i + 0.05 * i for i in range(len(rows))]
        seq = fresh(weight, bias)
        kwargs = dict(truncate=truncate, carry_hidden=carry,
                      initial_hidden=h0)
        if starts is not None:
            kwargs["segment_starts"] = starts
        if seeds is not None:
            kwargs["segment_hiddens"] = seeds
        seq.forward(rows, **kwargs)
        start_set = {i for i, flag in enumerate(starts or []) if flag}
        frozen = frozen_boundaries(weight, bias, rows, truncate, carry, h0,
                                   start_set, seeds)
        input_grads, grad_init, boundary_grads = \
            seq.backward_with_boundaries(go, grad_hidden)

        def obj(w=weight, b=bias, r=rows, h=h0, fr=frozen):
            return objective(w, b, r, h, go, grad_hidden, truncate, carry,
                             start_set, fr)

        # Input gradients, one entry per step (scalar for d == 1, per-feature
        # rows for d > 1).
        d = len(weight) - 1
        for t in range(len(rows)):
            for k in range(d):
                fd = central(lambda dlt, t=t, k=k: obj(
                    r=[row[:k] + [row[k] + dlt] + row[k + 1:]
                       if i == t else list(row)
                       for i, row in enumerate(rows)]))
                got = input_grads[t] if d == 1 else input_grads[t][k]
                self.assertTrue(close(got, fd),
                                "input grad (%d, %d): %r vs %r"
                                % (t, k, got, fd))
        # Parameter gradients accumulated on the Linear.
        for k in range(len(weight)):
            fd = central(lambda dlt, k=k: obj(
                w=[wi + (dlt if j == k else 0.0)
                   for j, wi in enumerate(weight)]))
            self.assertTrue(close(seq.linear.grad[k], fd),
                            "weight grad %d: %r vs %r"
                            % (k, seq.linear.grad[k], fd))
        fd_bias = central(lambda dlt: obj(b=bias + dlt))
        self.assertTrue(close(seq.linear.grad_bias, fd_bias))
        # Gradient with respect to the forward's initial_hidden. When an
        # explicit seed occupies position 0, h0 is detached and unused.
        fd_h0 = central(lambda dlt: obj(h=h0 + dlt))
        self.assertTrue(close(grad_init, fd_h0),
                        "grad_initial_hidden: %r vs %r" % (grad_init, fd_h0))
        # Boundary gradients, ascending index order, index 0 excluded.
        want_idx = sorted(i for i in boundary_indices(
            len(rows), truncate, start_set) if i > 0)
        self.assertEqual([i for i, _ in boundary_grads], want_idx)
        for (i, g) in boundary_grads:
            fd = central(lambda dlt, i=i: obj(
                fr={**frozen, i: frozen[i] + dlt}))
            self.assertTrue(close(g, fd),
                            "boundary grad %d: %r vs %r" % (i, g, fd))

    def test_grads_match_finite_differences_untruncated(self):
        self._check_pass(W, B, ROWS, 0.0, None, False, 0.0)

    def test_grads_match_finite_differences_across_all_branches(self):
        # Every recorded step lies in a different branch configuration; the
        # boundary cut severs the recurrent gradient at index 2.
        self._check_pass(WB, BB, BROWS, 0.0, None, False, 0.2)
        self._check_pass(WB, BB, BROWS, 0.1, 2, False, 0.1)
        self._check_pass(WB, BB, BROWS, 0.1, 2, True, 0.1)

    def test_grads_match_finite_differences_with_segment_starts(self):
        starts = [False, True, False, True, False]
        self._check_pass(W, B, ROWS, -0.1, 3, True, 0.05, starts)

    def test_grads_match_finite_differences_with_segment_hiddens(self):
        # A detached seed at the truncate boundary 2 (position 0 keeps the
        # plain h0 so the initial-hidden finite difference stays meaningful).
        seeds = [None, None, -0.25, None, None]
        self._check_pass(W, B, ROWS, 0.9, 2, True, 0.3, seeds=seeds)

    def test_grads_match_finite_differences_multi_feature(self):
        weight, bias = [1.5, -1.0, 0.6], 0.4
        rows = [[1.5, -0.4], [-2.0, 1.1], [0.3, 0.2]]
        self._check_pass(weight, bias, rows, 0.0, 2, True, 0.2)

    def test_backward_returns_only_input_grads(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        got = seq.backward(GO)
        self.assertIsInstance(got, list)
        self.assertTrue(all(isinstance(v, float) for v in got))
        ref = fresh()
        ref.forward(ROWS, truncate=2)
        ref.backward_with_boundaries(GO, 0.0)
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad, atol=1e-15))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias,
                              atol=1e-15))

    def test_multi_feature_input_grads_are_rows(self):
        seq = fresh([0.4, -0.3, 0.2], 0.1)
        seq.forward([[0.5, 0.1], [-0.2, 0.7]])
        got = seq.backward([0.3, -0.6])
        self.assertTrue(all(isinstance(row, list) and len(row) == 2
                            for row in got))

    def test_repeated_backward_accumulates(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        grad_once = list(seq.linear.grad)
        grad_bias_once = seq.linear.grad_bias
        seq.backward(GO)
        self.assertTrue(allclose(
            seq.linear.grad, [2.0 * g for g in grad_once], atol=1e-15))
        self.assertTrue(close(seq.linear.grad_bias, 2.0 * grad_bias_once,
                              atol=1e-15))
        ref = fresh()
        ref.forward(ROWS, truncate=2)
        ref.backward([2.0 * g for g in GO])
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad, atol=1e-15))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias,
                              atol=1e-15))

    def test_empty_sequence_backward(self):
        seq = fresh()
        seq.forward([], initial_hidden=0.3)
        self.assertEqual(seq.backward([]), [])
        input_grads, grad_init = seq.backward_with_initial_hidden([], 0.4)
        self.assertEqual(input_grads, [])
        self.assertEqual(grad_init, 0.4)
        _, _, boundary_grads = seq.backward_with_boundaries([], 0.4)
        self.assertEqual(boundary_grads, [])
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)

    def test_local_derivative_at_known_preactivation(self):
        # One step, unit input weight, zero bias and hidden weight: z = x.
        for x in (-4.0, -2.5, -1.0, 0.0, 2.5, 4.0):
            seq = HardSigmoidSequence(Linear([1.0, 0.0], 0.0))
            out = seq.forward([[x]])[0]
            self.assertEqual(out, hardsigmoid(x))
            seq.backward([0.8])
            want = 0.8 * hardsigmoid_derivative(x)
            self.assertTrue(close(seq.linear.grad[0], want * x, atol=1e-15))
            self.assertTrue(close(seq.linear.grad_bias, want, atol=1e-15))

    def test_runtime_errors(self):
        seq = fresh()
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        with self.assertRaises(RuntimeError):
            seq.backward_with_boundaries(GO)
        seq.start_stream()
        seq.step([0.2])
        with self.assertRaises(RuntimeError):
            seq.backward([0.1])
        seq.finish_stream()
        seq.backward([0.1])


class StreamTest(unittest.TestCase):
    def test_stream_matches_batch_forward_and_backward(self):
        for truncate, carry, h0 in ((None, False, 0.0), (2, True, 0.25),
                                    (3, False, -0.1)):
            batch = fresh()
            batch.forward(ROWS, truncate=truncate, carry_hidden=carry,
                          initial_hidden=h0)
            stream = fresh()
            stream.start_stream(initial_hidden=h0, truncate=truncate,
                                carry_hidden=carry)
            got = [stream.step(list(row)) for row in ROWS]
            self.assertTrue(allclose(got, batch.outputs, atol=1e-15))
            self.assertEqual(stream.finish_stream(), stream.outputs)
            self.assertTrue(allclose(stream.outputs, batch.outputs,
                                     atol=1e-15))
            gb = batch.backward_with_boundaries(GO, 0.1)
            gs = stream.backward_with_boundaries(GO, 0.1)
            self.assertTrue(allclose(gb[0], gs[0], atol=1e-15))
            self.assertTrue(close(gb[1], gs[1], atol=1e-15))
            self.assertEqual([i for i, _ in gb[2]], [i for i, _ in gs[2]])
            self.assertTrue(allclose([g for _, g in gb[2]],
                                     [g for _, g in gs[2]], atol=1e-15))
            self.assertTrue(allclose(batch.linear.grad, stream.linear.grad,
                                     atol=1e-15))

    def test_stream_matches_batch_branch_crossing(self):
        batch = fresh(WB, BB)
        batch.forward(BROWS, truncate=2, carry_hidden=True,
                      initial_hidden=0.2)
        stream = fresh(WB, BB)
        stream.start_stream(initial_hidden=0.2, truncate=2,
                            carry_hidden=True)
        for row in BROWS:
            stream.step(list(row))
        self.assertEqual(stream.finish_stream(), batch.outputs)
        go = [0.2, -0.4, 0.7, 0.1, -0.3]
        self.assertEqual(stream.backward(go), batch.backward(go))

    def test_stream_segment_start_marks_merge_with_truncate(self):
        stream = fresh()
        stream.start_stream(initial_hidden=0.1, truncate=3,
                            carry_hidden=True)
        stream.step(ROWS[0])
        stream.step(ROWS[1], segment_start=True)
        for row in ROWS[2:]:
            stream.step(list(row))
        stream.finish_stream()
        batch = fresh()
        batch.forward(ROWS, truncate=3, carry_hidden=True, initial_hidden=0.1,
                      segment_starts=[False, True, False, False, False])
        self.assertTrue(allclose(stream.outputs, batch.outputs, atol=1e-15))
        self.assertTrue(allclose(stream.backward(GO), batch.backward(GO),
                                 atol=1e-15))

    def test_stream_segment_hiddens_match_batch(self):
        seeds = [0.35, None, -0.25, None, None]
        batch = fresh()
        batch.forward(ROWS, truncate=2, carry_hidden=True,
                      segment_hiddens=seeds)
        stream = fresh()
        stream.start_stream(truncate=2, carry_hidden=True)
        for i, row in enumerate(ROWS):
            stream.step(list(row),
                        segment_hidden=seeds[i] if seeds[i] is not None
                        else None)
        self.assertEqual(stream.finish_stream(), batch.outputs)
        self.assertEqual(stream.backward(GO), batch.backward(GO))
        _, gi_b, bg_b = batch.backward_with_boundaries(GO, 0.1)
        _, gi_s, bg_s = stream.backward_with_boundaries(GO, 0.1)
        self.assertEqual(gi_b, gi_s)
        self.assertEqual(bg_b, bg_s)
        # A seed at a non-boundary step is rejected.
        stream2 = fresh()
        stream2.start_stream(truncate=2)
        stream2.step(ROWS[0])
        expect_value_error(lambda: stream2.step(ROWS[1],
                                                segment_hidden=0.5))

    def test_runtime_errors(self):
        seq = fresh()
        with self.assertRaises(RuntimeError):
            seq.finish_stream()
        with self.assertRaises(RuntimeError):
            seq.step([0.1], segment_start=True)
        with self.assertRaises(RuntimeError):
            seq.step([0.1], segment_hidden=0.2)
        seq.start_stream()
        with self.assertRaises(RuntimeError):
            seq.start_stream()
        seq.step([0.2])
        with self.assertRaises(RuntimeError):
            seq.backward([0.1])
        seq.finish_stream()
        seq.backward([0.1])


class CheckpointRestoreTest(unittest.TestCase):
    def test_checkpoint_restore_round_trip(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        seq.backward(GO)
        cp = seq.checkpoint()
        snap = seq.export_state()
        seq.step([9.9])
        seq.linear.apply_gradients(0.5)
        seq.start_stream()
        seq.step([0.1])
        self.assertIsNone(seq.restore(cp))
        self.assertEqual(seq.export_state(), snap)
        seq.step([0.4])
        seq.restore(cp)
        self.assertEqual(seq.export_state(), snap)
        # The restored cache still back-propagates identically.
        twin = fresh()
        twin.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        twin.backward(GO)
        self.assertEqual(seq.backward(GO), twin.backward(GO))

    def test_checkpoint_is_instance_bound(self):
        src = fresh()
        src.forward(ROWS)
        cp = src.checkpoint()
        other = fresh()
        expect_value_error(lambda: other.restore(cp))
        # Instances of the other sequence classes reject it (and it rejects
        # theirs), including the other z-recording classes whose checkpoints
        # carry the same extra-slot shape.
        for cls in OTHER_CLASSES:
            foreign = cls(Linear(list(W), B))
            expect_value_error(lambda foreign=foreign: foreign.restore(cp))
            expect_value_error(lambda foreign=foreign:
                               src.restore(foreign.checkpoint()))
        expect_value_error(lambda: src.restore("not a checkpoint"))

    def test_checkpoint_covers_open_stream(self):
        seq = fresh()
        seq.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        for row in ROWS[:3]:
            seq.step(list(row))
        cp = seq.checkpoint()
        snap = seq.export_state()
        for row in ROWS[3:]:
            seq.step(list(row))
        seq.restore(cp)
        self.assertEqual(seq.export_state(), snap)
        # The restored session continues exactly as the original would, and
        # its recorded pre-activations back-propagate identically.
        twin = fresh()
        twin.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        for row in ROWS[:3]:
            twin.step(list(row))
        for row in ROWS[3:]:
            self.assertEqual(seq.step(list(row)), twin.step(list(row)))
        self.assertEqual(seq.finish_stream(), twin.finish_stream())
        self.assertEqual(seq.backward(GO), twin.backward(GO))

    def test_corrupted_checkpoint_is_rejected(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        cp = seq.checkpoint()
        snap = seq.export_state()
        # Pre-activation slots must be lists of finite numbers matching the
        # trajectory length.
        for tamper in (
                lambda c: setattr(c, "_pres_fwd", None),
                lambda c: setattr(c, "_pres_fwd", [0.0]),
                lambda c: setattr(c, "_pres_fwd", [NAN] * len(ROWS)),
                lambda c: setattr(c, "_pres_fwd", [True] * len(ROWS)),
                lambda c: setattr(c, "_pres_fwd", "nope"),
                lambda c: setattr(c, "_pres_stream", []),
        ):
            broken = seq.checkpoint()
            tamper(broken)
            expect_value_error(lambda broken=broken: seq.restore(broken))
            self.assertEqual(seq.export_state(), snap)
        self.assertIsNone(seq.restore(cp))


class ExportImportTest(unittest.TestCase):
    def test_kind_and_version(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        state = seq.export_state()
        self.assertEqual(state["kind"], "HardSigmoidSequenceState")
        self.assertEqual(state["version"], 2)
        for cls in OTHER_CLASSES:
            self.assertNotEqual(state["kind"],
                                cls(Linear(list(W), B)).export_state()["kind"])
        self.assertEqual(json.loads(json.dumps(state)), state)
        self.assertIn("pre_activations", state["forward"])
        self.assertEqual(len(state["forward"]["pre_activations"]), len(ROWS))

    def test_pre_activations_match_trajectory(self):
        seq = fresh(WB, BB)
        seq.forward(BROWS, truncate=2, carry_hidden=True,
                    initial_hidden=0.2)
        state = seq.export_state()
        pres = state["forward"]["pre_activations"]
        # Independently recompute the pre-activations and verify each one
        # selects the branch its output shows.
        outs = state["forward"]["outputs"]
        prev = state["forward"]["prev_hiddens"]
        for z, y, h, row in zip(pres, outs, prev, BROWS):
            self.assertEqual(z, WB[0] * row[0] + WB[1] * h + BB)
            self.assertEqual(y, hardsigmoid(z))

    def test_migration_continues_identically(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2,
                    segment_starts=[False, True, False, False, False])
        src.backward_with_initial_hidden(GO, 0.1)
        state = src.export_state()

        dst = fresh([9.0, -9.0], 9.0)
        dst.forward([[0.1], [0.2]])
        dst.backward([1.0, 1.0])
        self.assertIsNone(dst.import_state(json.loads(json.dumps(state))))
        self.assertEqual(dst.export_state(), src.export_state())
        self.assertEqual(dst.backward(GO), src.backward(GO))
        self.assertEqual(dst.backward_with_boundaries(GO, 0.1),
                         src.backward_with_boundaries(GO, 0.1))
        self.assertEqual(dst.linear.grad, src.linear.grad)
        self.assertEqual(dst.step([0.6]), src.step([0.6]))
        self.assertEqual(dst.export_state(), src.export_state())

    def test_open_stream_migration_continues_identically(self):
        src = fresh()
        src.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        for row in ROWS[:3]:
            src.step(list(row))
        state = src.export_state()
        self.assertEqual(state["stream"]["initial_hidden"], 0.25)
        self.assertIsNone(state["forward"])

        dst = fresh()
        dst.import_state(copy.deepcopy(state))
        self.assertEqual(dst.export_state(), src.export_state())
        for row in ROWS[3:]:
            self.assertEqual(dst.step(list(row)), src.step(list(row)))
        self.assertEqual(dst.finish_stream(), src.finish_stream())
        self.assertEqual(dst.backward_with_boundaries(GO, 0.05),
                         src.backward_with_boundaries(GO, 0.05))

    def test_version_1_imports_missing_seeds_but_keeps_pre_activations(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        src.backward(GO)
        state = json.loads(json.dumps(src.export_state()))
        state["version"] = 1
        del state["forward"]["segment_hiddens"]
        dst = fresh()
        dst.import_state(state)
        # Missing v1 seeds are interpreted as None and normalized on commit.
        committed = dst.export_state()
        self.assertEqual(committed["forward"]["segment_hiddens"],
                         [None] * len(ROWS))
        self.assertEqual(committed["forward"]["pre_activations"],
                         src.export_state()["forward"]["pre_activations"])
        self.assertEqual(dst.backward(GO), src.backward(GO))
        # Even a v1 record must carry its pre-activations: HardSigmoid cannot
        # differentiate without them.
        no_pres = copy.deepcopy(state)
        del no_pres["forward"]["pre_activations"]
        snap = dst.export_state()
        expect_value_error(lambda: dst.import_state(no_pres))
        self.assertEqual(dst.export_state(), snap)

    def test_cross_kind_states_are_rejected(self):
        hs_seq = fresh()
        hs_seq.forward(ROWS, truncate=2)
        others = [cls(Linear(list(W), B)) for cls in OTHER_CLASSES]
        for other in others:
            other.forward(ROWS, truncate=2)
        hs_snap = hs_seq.export_state()
        other_snaps = [other.export_state() for other in others]
        other_states = [other.export_state() for other in others]
        for other, other_state in zip(others, other_states):
            expect_value_error(
                lambda other_state=other_state:
                hs_seq.import_state(other_state))
            expect_value_error(
                lambda other=other: other.import_state(hs_snap))
        self.assertEqual(hs_seq.export_state(), hs_snap)
        for other, other_snap in zip(others, other_snaps):
            self.assertEqual(other.export_state(), other_snap)

    def test_rejected_import_leaves_state_untouched(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        src.backward(GO)
        dst = fresh()
        dst.forward([[0.1], [0.2]])
        snap = dst.export_state()
        for tamper in (
                lambda s: s.update(version=3),
                lambda s: s.update(width=3),
                lambda s: s.update(hidden=NAN),
                lambda s: s["forward"].update(truncate=0),
                lambda s: s["linear"].update(last=[1.0]),
                lambda s: s.update(kind="TanhSequenceState"),
                lambda s: s.update(kind="HardSwishSequenceState"),
                lambda s: s["forward"].update(pre_activations=[0.0]),
                lambda s: s["forward"].update(
                    pre_activations=[NAN] * len(ROWS)),
                lambda s: s["forward"].pop("pre_activations"),
                lambda s: s["forward"]["pre_activations"].__setitem__(0, INF),
        ):
            state = copy.deepcopy(src.export_state())
            tamper(state)
            expect_value_error(lambda state=state: dst.import_state(state))
            self.assertEqual(dst.export_state(), snap)


class ValidationTest(unittest.TestCase):
    def test_forward_argument_validation(self):
        seq = fresh()
        for bad in (0, -1, 1.5, True, "2"):
            expect_value_error(
                lambda bad=bad: seq.forward(ROWS, truncate=bad))
        expect_value_error(lambda: seq.forward(ROWS, carry_hidden=1))
        for bad in (NAN, INF, -INF, BIG, "0.1", True):
            expect_value_error(
                lambda bad=bad: seq.forward(ROWS, initial_hidden=bad))
        expect_value_error(lambda: seq.forward(
            ROWS, segment_starts=[True] * (len(ROWS) - 1)))
        expect_value_error(lambda: seq.forward(
            ROWS, segment_starts=[True] * (len(ROWS) - 1) + [1]))
        for bad_row in ([0.1, 0.2], [], [True], ["0.1"], [NAN], [INF], [BIG]):
            expect_value_error(
                lambda bad_row=bad_row: fresh().forward([bad_row]))
        expect_value_error(lambda: fresh().forward([[0.1], [NAN]]))
        for bad_seeds in ("x", [0.1] * 4, [True] * 5, [[None] * 5]):
            expect_value_error(
                lambda bad_seeds=bad_seeds:
                seq.forward(ROWS, truncate=2, segment_hiddens=bad_seeds))

    def test_backward_argument_validation(self):
        seq = fresh()
        seq.forward(ROWS)
        grad, bias_grad = list(seq.linear.grad), seq.linear.grad_bias
        for bad in (GO[:-1], GO + [0.1], [NAN] * len(ROWS), [True] * 5,
                    "grads"):
            expect_value_error(lambda bad=bad: seq.backward(bad))
        for bad in (NAN, INF, BIG, True, "0.1"):
            expect_value_error(
                lambda bad=bad: seq.backward_with_initial_hidden(GO, bad))
        self.assertEqual(seq.linear.grad, grad)
        self.assertEqual(seq.linear.grad_bias, bias_grad)

    def test_stream_argument_validation(self):
        seq = fresh()
        for bad in (0, -1, 1.5, True):
            expect_value_error(lambda bad=bad: seq.start_stream(truncate=bad))
        expect_value_error(lambda: seq.start_stream(carry_hidden=1))
        expect_value_error(lambda: seq.start_stream(initial_hidden=NAN))
        seq.start_stream()
        for bad_row in ([0.1, 0.2], [NAN], [True], ["0.1"]):
            expect_value_error(lambda bad_row=bad_row: seq.step(bad_row))
        expect_value_error(lambda: seq.step([0.1], segment_start=1))
        expect_value_error(lambda: seq.step([0.1], segment_hidden=NAN))
        seq.finish_stream()


class AtomicityTest(unittest.TestCase):
    def snapshot(self, seq):
        return {
            "hidden": seq.hidden,
            "outputs": list(seq.outputs),
            "fwd": copy.deepcopy(seq._fwd),
            "stream": copy.deepcopy(seq._stream),
            "weight": list(seq.linear.weight),
            "bias": seq.linear.bias,
            "grad": list(seq.linear.grad),
            "grad_bias": seq.linear.grad_bias,
            "last": None if seq.linear.last is None else list(seq.linear.last),
            "last_weight": None if seq.linear._last_weight is None
            else list(seq.linear._last_weight),
        }

    def test_nonfinite_forward_leaves_everything_untouched(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        # A finite, legal row whose linear arithmetic overflows: the
        # non-finite computation fails with ValueError and commits nothing.
        seq.linear.weight = [1e308, 0.5]
        before = self.snapshot(seq)
        expect_value_error(lambda: seq.forward([[10.0]]))
        self.assertEqual(self.snapshot(seq), before)
        # The earlier cache (with its recorded z values) still back-propagates.
        seq.backward(GO)

    def test_nonfinite_stream_step_leaves_session_untouched(self):
        seq = fresh()
        seq.start_stream(initial_hidden=0.1, truncate=2)
        seq.step([0.8])
        seq.step([-0.5])
        seq.linear.weight = [1e308, 0.5]
        before = self.snapshot(seq)
        expect_value_error(lambda: seq.step([10.0], segment_start=True))
        self.assertEqual(self.snapshot(seq), before)
        seq.linear.weight = [0.4, -0.3]
        seq.step([10.0], segment_start=True)
        self.assertEqual(len(seq.finish_stream()), 3)

    def test_overflowing_backward_accumulates_nothing(self):
        seq = fresh()
        seq.forward(ROWS)
        seq.backward(GO)
        before = self.snapshot(seq)
        expect_value_error(
            lambda: seq.backward_with_initial_hidden([1e308] * len(ROWS),
                                                     1e308))
        self.assertEqual(self.snapshot(seq), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
