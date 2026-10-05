"""GELUSequence: the Gaussian-error-linear-unit sequence cell.

Mirrors the TanhSequence/SigmoidSequence/SoftplusSequence/ReLUSequence/
LeakyReLUSequence conventions: independent erf-based reference formulas,
finite-difference gradient checks, stream/batch value equality, checkpoint
and state-migration rules (with a kind tag distinct from every other
sequence class), validation and failure atomicity. The constructor accepts
only a Linear.
"""
import copy
import json
import math
import unittest

import app
from app import (GELUSequence, LeakyReLUSequence, Linear, ReLUSequence,
                 SigmoidSequence, SoftplusSequence, TanhSequence)

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400  # finite int, but overflows double arithmetic

EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-7

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)

W, B = [0.4, -0.3], 0.15
ROWS = [[0.8], [-0.5], [1.2], [-0.7], [0.3]]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]


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

def gelu(z):
    return 0.5 * z * (1.0 + math.erf(z / SQRT_2))


def gelu_prime(z):
    return 0.5 * (1.0 + math.erf(z / SQRT_2)) \
        + z * math.exp(-0.5 * z * z) / SQRT_2PI


def sequence_outputs(weight, bias, rows, h0, truncate, carry, starts=(),
                     frozen=None):
    """gelu(w_in.row + w_h*h_prev + b), with the published boundary rules.

    ``starts`` holds explicit segment-start indices (the segment_starts
    argument); ``frozen`` maps a segment-start index past 0 to the detached
    constant hidden value consumed there.
    """
    outs = []
    hidden = h0
    for i, row in enumerate(rows):
        if (truncate is not None and i % truncate == 0) or i in starts:
            if i == 0:
                hidden = h0
            elif frozen is not None:
                hidden = frozen[i]
            elif carry:
                hidden = outs[-1]
            else:
                hidden = 0.0
        z = sum(w * a for w, a in zip(weight[:-1], row)) \
            + weight[-1] * hidden + bias
        hidden = gelu(z)
        outs.append(hidden)
    return outs


def boundary_indices(n, truncate, starts=()):
    return {i for i in range(n)
            if (truncate is not None and i % truncate == 0) or i in starts}


def frozen_boundaries(rows, truncate, carry, weight, bias, h0, starts=()):
    """Boundary constants taken from the unperturbed trajectory."""
    base = sequence_outputs(weight, bias, rows, h0, truncate, carry, starts)
    frozen = {}
    for i in boundary_indices(len(rows), truncate, starts):
        if i > 0:
            frozen[i] = base[i - 1] if carry else 0.0
    return frozen


def objective(weight, bias, rows, h0, grad_outputs, grad_hidden,
              truncate, carry, starts=(), frozen=None):
    """L = sum_t go_t * out_t + grad_hidden * out_{-1} (grad_hidden * h0 when
    the sequence is empty)."""
    outs = sequence_outputs(weight, bias, rows, h0, truncate, carry,
                            starts, frozen)
    if not outs:
        return grad_hidden * h0
    return sum(g * o for g, o in zip(grad_outputs, outs)) \
        + grad_hidden * outs[-1]


def fresh(weight=W, bias=B):
    return GELUSequence(Linear(list(weight), bias))


OTHER_CLASSES = (TanhSequence, SigmoidSequence, SoftplusSequence,
                 ReLUSequence, LeakyReLUSequence)


class SmokeTest(unittest.TestCase):
    def test_module_exposes_gelu_sequence(self):
        self.assertTrue(hasattr(app, "GELUSequence"))
        self.assertTrue(issubclass(GELUSequence, TanhSequence))

    def test_construction_rules_match_tanh(self):
        expect_value_error(lambda: GELUSequence("not a linear"))
        expect_value_error(lambda: GELUSequence(Linear([1.0])))
        seq = GELUSequence(Linear([0.4, -0.3], 0.15))
        self.assertEqual(seq.d, 1)
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])

    def test_no_extra_constructor_arguments(self):
        # The constructor takes only the Linear, exactly like the others.
        with self.assertRaises(TypeError):
            GELUSequence(Linear([0.4, -0.3], 0.15), 0.2)
        with self.assertRaises(TypeError):
            GELUSequence(Linear([0.4, -0.3], 0.15), approximate=True)

    def test_activation_and_derivative_are_finite_functions(self):
        for z in (-1e308, -1e200, -1e12, -3.0, -0.5, -0.0, 0.0, 0.5, 3.0,
                  1e12, 1e200, 1e308):
            out = GELUSequence._activate(z)
            der = GELUSequence._activation_derivative(z)
            self.assertTrue(math.isfinite(out), (z, out))
            self.assertTrue(math.isfinite(der), (z, der))


class ForwardTest(unittest.TestCase):
    def test_forward_matches_reference_recurrence(self):
        for truncate, carry, h0 in ((None, False, 0.0), (2, False, 0.0),
                                    (2, True, 0.2), (3, False, -0.4),
                                    (1, True, 0.0)):
            seq = fresh()
            got = seq.forward(ROWS, truncate=truncate, carry_hidden=carry,
                              initial_hidden=h0)
            want = sequence_outputs(W, B, ROWS, h0, truncate, carry)
            self.assertTrue(allclose(got, want, atol=1e-14),
                            "%r vs %r" % (got, want))
            self.assertTrue(close(seq.hidden, want[-1], atol=1e-14))
            self.assertEqual(seq.outputs, got)

    def test_forward_with_segment_starts_merges_boundaries(self):
        starts = [False, True, False, False, True]
        for carry in (False, True):
            seq = fresh()
            got = seq.forward(ROWS, truncate=3, carry_hidden=carry,
                              initial_hidden=0.2, segment_starts=starts)
            want = sequence_outputs(W, B, ROWS, 0.2, 3, carry,
                                    starts={1, 4})
            self.assertTrue(allclose(got, want, atol=1e-14))

    def test_multi_feature_forward_matches_reference(self):
        weight, bias = [0.4, -0.3, 0.2], 0.1
        rows = [[0.5, 0.1], [-0.2, 0.7], [0.9, -0.4]]
        seq = fresh(weight, bias)
        got = seq.forward(rows, truncate=2)
        want = sequence_outputs(weight, bias, rows, 0.0, 2, False)
        self.assertTrue(allclose(got, want, atol=1e-14))

    def test_empty_forward_commits_initial_hidden(self):
        seq = fresh()
        self.assertEqual(seq.forward([], initial_hidden=0.3), [])
        self.assertEqual(seq.hidden, 0.3)
        self.assertEqual(seq.outputs, [])
        self.assertEqual(seq._fwd_pre, [])

    def test_extreme_preactivations_stay_finite_without_overflow(self):
        # |z| around 1e12: a huge positive z passes through as exactly z
        # (erf saturated to 1), a huge negative z yields a finite zero (the
        # 1 + erf tail rounds to a true zero, so the product cannot be
        # inf); no exp/multiply overflow is possible.
        seq = GELUSequence(Linear([1e6, 0.0], 0.0))
        got = seq.forward([[1e6], [-1e6], [1e6]])
        self.assertEqual(got[0], 1e12)
        self.assertEqual(got[2], 1e12)
        self.assertEqual(got[1], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in got))
        # step() shares the same stability guarantee.
        self.assertEqual(seq.step([1e6]), 1e12)
        neg = seq.step([-1e6])
        self.assertEqual(neg, 0.0)
        self.assertTrue(math.isfinite(neg))

    def test_extreme_preactivations_near_double_range(self):
        # Even |z| close to the largest finite double must not overflow the
        # 0.5*z multiplication or the exp(-z*z/2) derivative term.
        for z in (1e308, -1e308):
            out = GELUSequence._activate(z)
            der = GELUSequence._activation_derivative(z)
            self.assertTrue(math.isfinite(out))
            self.assertTrue(math.isfinite(der))
        self.assertEqual(GELUSequence._activate(1e308), 1e308)
        self.assertEqual(GELUSequence._activate(-1e308), 0.0)
        self.assertEqual(GELUSequence._activation_derivative(1e308), 1.0)
        self.assertEqual(GELUSequence._activation_derivative(-1e308), 0.0)

    def test_zero_preactivation_outputs_exact_positive_zero(self):
        # z == 0 exactly: the output is exactly 0.0 with a positive sign,
        # including for a -0.0 pre-activation.
        seq = GELUSequence(Linear([1.0, 0.0], -1.0))
        got = seq.forward([[1.0]])
        self.assertEqual(got, [0.0])
        self.assertEqual(math.copysign(1.0, got[0]), 1.0)
        self.assertEqual(GELUSequence._activate(-0.0), 0.0)
        self.assertEqual(math.copysign(1.0, GELUSequence._activate(-0.0)), 1.0)

    def test_activation_keeps_formula_precision_near_zero(self):
        # gelu(z) = 0.5*z + z^2/sqrt(2pi) + O(z^4); a naive 0.5*z*(1+erf)
        # loses the z^2 term by cancellation. The erfc-based branch keeps
        # full precision.
        for z in (1e-6, 1e-8, 1e-10):
            got = GELUSequence._activate(z)
            series = 0.5 * z + z * z / SQRT_2PI
            self.assertTrue(abs(got - series) <= 1e-15 * (1.0 + abs(series)),
                            (z, got, series))
        # gelu'(z) = 0.5 + 2*z/sqrt(2pi) + O(z^3); a naive 1 + erf loses
        # the erf tail entirely around 1e-12 (error ~4e-13).
        for z in (1e-10, 1e-12):
            got = GELUSequence._activation_derivative(z)
            series = 0.5 + 2.0 * z / SQRT_2PI
            self.assertTrue(abs(got - series) <= 2e-16, (z, got, series))
        self.assertEqual(GELUSequence._activation_derivative(0.0), 0.5)


class BackwardTest(unittest.TestCase):
    def _check_pass(self, weight, bias, rows, h0, truncate, carry,
                    grad_hidden, starts=None):
        go = [0.3 * (-1) ** i + 0.05 * i for i in range(len(rows))]
        seq = fresh(weight, bias)
        kwargs = dict(truncate=truncate, carry_hidden=carry,
                      initial_hidden=h0)
        if starts is not None:
            kwargs["segment_starts"] = starts
        seq.forward(rows, **kwargs)
        start_set = {i for i, flag in enumerate(starts or []) if flag}
        frozen = frozen_boundaries(rows, truncate, carry, weight, bias, h0,
                                   start_set)
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
                self.assertTrue(close(got, fd, atol=1e-6),
                                "input grad (%d, %d): %r vs %r"
                                % (t, k, got, fd))
        # Parameter gradients accumulated on the Linear.
        for k in range(len(weight)):
            fd = central(lambda dlt, k=k: obj(
                w=[wi + (dlt if j == k else 0.0)
                   for j, wi in enumerate(weight)]))
            self.assertTrue(close(seq.linear.grad[k], fd, atol=1e-6),
                            "weight grad %d: %r vs %r"
                            % (k, seq.linear.grad[k], fd))
        fd_bias = central(lambda dlt: obj(b=bias + dlt))
        self.assertTrue(close(seq.linear.grad_bias, fd_bias, atol=1e-6))
        # Gradient with respect to the forward's initial_hidden.
        fd_h0 = central(lambda dlt: obj(h=h0 + dlt))
        self.assertTrue(close(grad_init, fd_h0, atol=1e-6),
                        "grad_initial_hidden: %r vs %r" % (grad_init, fd_h0))
        # Boundary gradients, ascending index order, index 0 excluded.
        want_idx = sorted(i for i in boundary_indices(
            len(rows), truncate, start_set) if i > 0)
        self.assertEqual([i for i, _ in boundary_grads], want_idx)
        for (i, g) in boundary_grads:
            fd = central(lambda dlt, i=i: obj(
                fr={**frozen, i: frozen[i] + dlt}))
            self.assertTrue(close(g, fd, atol=1e-6),
                            "boundary grad %d: %r vs %r" % (i, g, fd))

    def test_grads_match_finite_differences_untruncated(self):
        self._check_pass(W, B, ROWS, 0.0, None, False, 0.0)

    def test_grads_match_finite_differences_truncated(self):
        for carry in (False, True):
            self._check_pass(W, B, ROWS, 0.2, 2, carry, 0.1)

    def test_grads_match_finite_differences_with_segment_starts(self):
        starts = [False, True, False, True, False]
        self._check_pass(W, B, ROWS, -0.1, 3, True, 0.05, starts)

    def test_grads_match_finite_differences_multi_feature(self):
        weight, bias = [0.4, -0.3, 0.2], 0.1
        rows = [[0.5, 0.1], [-0.2, 0.7], [0.9, -0.4]]
        self._check_pass(weight, bias, rows, 0.0, 2, True, 0.2)

    def test_local_derivative_matches_published_formula(self):
        # One step, unit input weight: d_pre = go * gelu'(z), and the
        # accumulated input/weight/bias gradients follow directly.
        for z in (-1.7, -0.3, 0.0, 0.4, 1.9):
            seq = GELUSequence(Linear([1.0, 0.0], 0.0))
            out = seq.forward([[z]])[0]
            self.assertTrue(close(out, gelu(z), atol=1e-14))
            got = seq.backward([0.8])
            want = 0.8 * gelu_prime(z)
            self.assertTrue(close(got[0], want, atol=1e-14), (z, got[0], want))
            self.assertTrue(close(seq.linear.grad[0], want * z, atol=1e-14))
            self.assertTrue(close(seq.linear.grad[1], 0.0, atol=1e-14))
            self.assertTrue(close(seq.linear.grad_bias, want, atol=1e-14))

    def test_backward_returns_only_input_grads(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        got = seq.backward(GO)
        self.assertIsInstance(got, list)
        self.assertTrue(all(isinstance(v, float) for v in got))
        ref = fresh()
        ref.forward(ROWS, truncate=2)
        ref.backward_with_boundaries(GO, 0.0)
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad, atol=1e-14))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias,
                              atol=1e-14))

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
            seq.linear.grad, [2.0 * g for g in grad_once], atol=1e-14))
        self.assertTrue(close(seq.linear.grad_bias, 2.0 * grad_bias_once,
                              atol=1e-14))
        ref = fresh()
        ref.forward(ROWS, truncate=2)
        ref.backward([2.0 * g for g in GO])
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad, atol=1e-14))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias,
                              atol=1e-14))

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


class StreamTest(unittest.TestCase):
    def test_stream_matches_batch_forward_and_backward_value_for_value(self):
        for truncate, carry, h0 in ((None, False, 0.0), (2, True, 0.25),
                                    (3, False, -0.1), (1, True, 0.2)):
            batch = fresh()
            batch.forward(ROWS, truncate=truncate, carry_hidden=carry,
                          initial_hidden=h0)
            stream = fresh()
            stream.start_stream(initial_hidden=h0, truncate=truncate,
                                carry_hidden=carry)
            got = [stream.step(list(row)) for row in ROWS]
            # The row-wise session and the batch forward agree value for
            # value (exactly, not just approximately).
            self.assertEqual(got, batch.outputs)
            self.assertEqual(stream.finish_stream(), stream.outputs)
            self.assertEqual(stream.outputs, batch.outputs)
            gb = batch.backward_with_boundaries(GO, 0.1)
            gs = stream.backward_with_boundaries(GO, 0.1)
            self.assertEqual(gb[0], gs[0])
            self.assertEqual(gb[1], gs[1])
            self.assertEqual(gb[2], gs[2])
            self.assertEqual(batch.linear.grad, stream.linear.grad)
            self.assertEqual(batch.linear.grad_bias,
                             stream.linear.grad_bias)

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
        self.assertEqual(stream.outputs, batch.outputs)
        self.assertEqual(stream.backward(GO), batch.backward(GO))

    def test_runtime_errors(self):
        seq = fresh()
        with self.assertRaises(RuntimeError):
            seq.finish_stream()
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        with self.assertRaises(RuntimeError):
            seq.step([0.1], segment_start=True)
        seq.start_stream()
        with self.assertRaises(RuntimeError):
            seq.start_stream()
        seq.step([0.2])
        with self.assertRaises(RuntimeError):
            seq.backward([0.1])
        seq.finish_stream()
        seq.backward([0.1])
        # A standalone step after a cached batch pass invalidates the cache.
        seq.forward(ROWS[:2])
        seq.step([0.2])
        with self.assertRaises(RuntimeError):
            seq.backward([0.1, 0.2])

    def test_failed_stream_step_rolls_back_pre_activations(self):
        seq = fresh()
        seq.start_stream(initial_hidden=0.1, truncate=2)
        seq.step([0.8])
        pres_before = list(seq._stream_pre)
        expect_value_error(lambda: seq.step([BIG]))
        self.assertEqual(seq._stream_pre, pres_before)
        self.assertEqual(len(seq._stream["_gelu_pre"]), 1)
        # The legal retry records exactly one additional pre-activation.
        seq.step([0.8])
        self.assertEqual(len(seq._stream_pre), 2)
        outputs = seq.finish_stream()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(seq._fwd_pre), 2)


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

    def test_checkpoint_carries_pre_activations(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        seq.backward(GO)
        input_grads = seq.backward(GO)  # second accumulation
        cp = seq.checkpoint()
        self.assertEqual(len(cp._fwd_pres), len(ROWS))
        # Diverge with an unrelated pass that replaces the cache and its
        # pre-activations, then restore: the original z list must return.
        seq.forward([[0.1], [0.2]])
        self.assertEqual(len(seq._fwd_pre), 2)
        seq.restore(cp)
        self.assertEqual(len(seq._fwd_pre), len(ROWS))
        # The restored pre-activations back-propagate to the same input
        # gradients (input gradients depend only on the cached pass).
        self.assertEqual(seq.backward(GO), input_grads)
        ref = fresh()
        ref.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        self.assertEqual(seq.backward(GO), ref.backward(GO))

    def test_checkpoint_is_instance_bound(self):
        src = fresh()
        src.forward(ROWS)
        cp = src.checkpoint()
        other = fresh()
        expect_value_error(lambda: other.restore(cp))
        # Instances of every other sequence class reject it, and GELU
        # rejects theirs.
        for cls in OTHER_CLASSES:
            foreign = cls(Linear(list(W), B))
            expect_value_error(lambda foreign=foreign: foreign.restore(cp))
            expect_value_error(lambda foreign=foreign:
                               src.restore(foreign.checkpoint()))
        expect_value_error(lambda: src.restore("not a checkpoint"))
        expect_value_error(lambda: src.restore(
            LeakyReLUSequence(Linear(list(W), B)).checkpoint()))

    def test_checkpoint_covers_open_stream(self):
        seq = fresh()
        seq.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        for row in ROWS[:3]:
            seq.step(list(row))
        cp = seq.checkpoint()
        snap = seq.export_state()
        self.assertEqual(len(cp._stream_pres), 3)
        for row in ROWS[3:]:
            seq.step(list(row))
        seq.restore(cp)
        self.assertEqual(seq.export_state(), snap)
        # The restored session continues exactly as the original would.
        twin = fresh()
        twin.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        for row in ROWS[:3]:
            twin.step(list(row))
        for row in ROWS[3:]:
            self.assertEqual(seq.step(list(row)), twin.step(list(row)))
        self.assertEqual(seq.finish_stream(), twin.finish_stream())
        self.assertEqual(len(seq._fwd_pre), len(ROWS))

    def test_corrupted_pre_activation_list_is_rejected_atomically(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        snap = seq.export_state()

        cp = seq.checkpoint()
        cp._fwd_pres.append(0.5)
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap)

        cp = seq.checkpoint()
        cp._fwd_pres[0] = NAN
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap)

        cp = seq.checkpoint()
        cp._fwd_pres = None
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap)

        seq.start_stream(initial_hidden=0.2, truncate=2)
        seq.step(ROWS[0])
        cp = seq.checkpoint()
        cp._stream_pres = "nope"
        before = seq.export_state()
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), before)


class ExportImportTest(unittest.TestCase):
    def test_kind_and_version(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        state = seq.export_state()
        self.assertEqual(state["kind"], "GELUSequenceState")
        self.assertEqual(state["version"], 1)
        for cls in OTHER_CLASSES:
            self.assertNotEqual(state["kind"],
                                cls(Linear(list(W), B)).export_state()["kind"])
        self.assertEqual(json.loads(json.dumps(state)), state)

    def test_pre_lists_align_with_trajectories(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        state = seq.export_state()
        self.assertEqual(len(state["forward_pre"]), len(ROWS))
        self.assertIsNone(state["stream_pre"])
        self.assertTrue(all(isinstance(v, float) and math.isfinite(v)
                            for v in state["forward_pre"]))
        seq.start_stream(initial_hidden=0.1)
        seq.step(ROWS[0])
        open_state = seq.export_state()
        self.assertIsNone(open_state["forward"])
        self.assertIsNone(open_state["forward_pre"])
        self.assertEqual(len(open_state["stream_pre"]), 1)

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
        self.assertEqual(len(state["stream_pre"]), 3)

        dst = fresh()
        dst.import_state(copy.deepcopy(state))
        self.assertEqual(dst.export_state(), src.export_state())
        for row in ROWS[3:]:
            self.assertEqual(dst.step(list(row)), src.step(list(row)))
        self.assertEqual(dst.finish_stream(), src.finish_stream())
        self.assertEqual(dst.backward_with_boundaries(GO, 0.05),
                         src.backward_with_boundaries(GO, 0.05))

    def test_cross_kind_states_are_rejected(self):
        gelu_seq = fresh()
        gelu_seq.forward(ROWS, truncate=2)
        others = [cls(Linear(list(W), B)) for cls in OTHER_CLASSES]
        for other in others:
            other.forward(ROWS, truncate=2)
        gelu_snap = gelu_seq.export_state()
        other_snaps = [other.export_state() for other in others]
        other_states = [other.export_state() for other in others]
        for other, other_state in zip(others, other_states):
            expect_value_error(
                lambda other_state=other_state:
                gelu_seq.import_state(other_state))
            expect_value_error(
                lambda other=other: other.import_state(gelu_snap))
        self.assertEqual(gelu_seq.export_state(), gelu_snap)
        for other, other_snap in zip(others, other_snaps):
            self.assertEqual(other.export_state(), other_snap)

    def test_width_mismatch_is_rejected(self):
        src = fresh([0.4, -0.3, 0.2], 0.1)
        src.forward([[0.5, 0.1], [-0.2, 0.7]])
        state = src.export_state()
        dst = fresh()
        dst.forward(ROWS[:1])
        snap = dst.export_state()
        expect_value_error(lambda: dst.import_state(state))
        self.assertEqual(dst.export_state(), snap)

    def test_rejected_import_leaves_state_untouched(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        src.backward(GO)
        dst = fresh()
        dst.forward([[0.1], [0.2]])
        snap = dst.export_state()

        def drop_pre(s):
            s["forward_pre"] = None

        def extra_pre(s):
            s["forward_pre"].append(1.0)

        def nonfinite_pre(s):
            s["forward_pre"][0] = NAN

        def pre_without_trajectory(s):
            s["forward"] = None

        def pre_wrong_type(s):
            s["stream_pre"] = [1.0]

        for tamper in (
                lambda s: s.update(version=2),
                lambda s: s.update(width=3),
                lambda s: s.update(hidden=NAN),
                lambda s: s["forward"].update(truncate=0),
                lambda s: s["linear"].update(last=[1.0]),
                lambda s: s.update(kind="ReLUSequenceState"),
                drop_pre,
                extra_pre,
                nonfinite_pre,
                pre_without_trajectory,
                pre_wrong_type,
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
        seq.finish_stream()


class AtomicityTest(unittest.TestCase):
    def snapshot(self, seq):
        return {
            "hidden": seq.hidden,
            "outputs": list(seq.outputs),
            "fwd": copy.deepcopy(seq._fwd),
            "stream": copy.deepcopy(seq._stream),
            "fwd_pre": None if seq._fwd_pre is None else list(seq._fwd_pre),
            "stream_pre": None if seq._stream_pre is None
            else list(seq._stream_pre),
            "weight": list(seq.linear.weight),
            "bias": seq.linear.bias,
            "grad": list(seq.linear.grad),
            "grad_bias": seq.linear.grad_bias,
            "last": None if seq.linear.last is None else list(seq.linear.last),
            "last_weight": None if seq.linear._last_weight is None
            else list(seq.linear._last_weight),
        }

    def test_overflowing_forward_leaves_everything_untouched(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        seq.linear.weight = [1e308, 0.5]
        before = self.snapshot(seq)
        expect_value_error(lambda: seq.forward([[10.0]]))
        self.assertEqual(self.snapshot(seq), before)
        # The earlier cache (with its pre-activations) still backprops.
        seq.backward(GO)

    def test_overflowing_stream_step_leaves_session_untouched(self):
        seq = GELUSequence(Linear([0.4, -0.3], 0.15))
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
        # The cache and pre-activations survive for a valid retry.
        self.assertTrue(all(math.isfinite(v) for v in seq.backward(GO)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
