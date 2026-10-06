"""SELUSequence: the scaled exponential linear unit (self-normalizing).

Mirrors the TanhSequence/.../ELUSequence conventions: independent reference
formulas, finite-difference gradient checks, stream/batch equivalence,
checkpoint and state-migration rules (with the kind tag
"SELUSequenceState", distinct from every other sequence class), validation
and failure atomicity. alpha and scale are fixed (1.6732632423543772 and
1.0507009873554805); the constructor accepts only a Linear.
"""
import copy
import json
import math
import unittest

import app
from app import (ELUSequence, GELUSequence, LeakyReLUSequence, Linear,
                 MishSequence, ReLUSequence, SELUSequence, SigmoidSequence,
                 SiLUSequence, SoftplusSequence, SoftsignSequence,
                 TanhSequence)

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400  # finite int, but overflows double arithmetic
# A finite double whose product with the SELU scale overflows (z*scale is
# about 1.84e308 > DBL_MAX, while z itself is finite).
OVERFLOWING_POSITIVE_Z = 1.75e308

ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805
SCALE_ALPHA = SCALE * ALPHA

EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-7

W, B = [0.4, -0.3], 0.15
ROWS = [[0.8], [-0.5], [1.2], [-0.7], [0.3]]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]

OTHER_CLASSES = (TanhSequence, SigmoidSequence, SoftplusSequence,
                 ReLUSequence, LeakyReLUSequence, ELUSequence, GELUSequence,
                 SiLUSequence, MishSequence, SoftsignSequence)


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

def selu(z):
    if z > 0:
        return SCALE * z
    if z == 0:
        return 0.0
    return SCALE_ALPHA * math.expm1(z)


def selu_deriv(z):
    return SCALE if z > 0 else SCALE_ALPHA * math.exp(z)


def sequence_outputs(weight, bias, rows, h0, truncate, carry, starts=(),
                     frozen=None):
    """selu(w_in.row + w_h*h_prev + b), with the published boundary rules.

    ``starts`` holds explicit segment-start indices (the segment_starts
    argument); ``frozen`` maps a segment-start index past 0 to the detached
    constant hidden value consumed there.
    """
    outs = []
    hidden = h0
    for i, row in enumerate(rows):
        if (truncate is not None and i % truncate == 0) or i in starts:
            if frozen is not None and i in frozen:
                # An explicit detached seed takes precedence over every
                # other rule, including h0 at position 0.
                hidden = frozen[i]
            elif i == 0:
                hidden = h0
            elif carry:
                hidden = outs[-1]
            else:
                hidden = 0.0
        z = sum(w * a for w, a in zip(weight[:-1], row)) \
            + weight[-1] * hidden + bias
        hidden = selu(z)
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
    return SELUSequence(Linear(list(weight), bias))


class SmokeTest(unittest.TestCase):
    def test_module_exposes_selu_sequence(self):
        self.assertTrue(hasattr(app, "SELUSequence"))
        self.assertTrue(issubclass(SELUSequence, TanhSequence))

    def test_construction_rules_match_tanh(self):
        expect_value_error(lambda: SELUSequence("not a linear"))
        expect_value_error(lambda: SELUSequence(Linear([1.0])))
        seq = SELUSequence(Linear([0.4, -0.3], 0.15))
        self.assertEqual(seq.d, 1)
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])

    def test_no_constant_configuration(self):
        # The constructor takes only the Linear; alpha and scale are fixed
        # and cannot be passed positionally or by keyword.
        with self.assertRaises(TypeError):
            SELUSequence(Linear([0.4, -0.3], 0.15), 0.2)
        with self.assertRaises(TypeError):
            SELUSequence(Linear([0.4, -0.3], 0.15), alpha=0.2)
        with self.assertRaises(TypeError):
            SELUSequence(Linear([0.4, -0.3], 0.15), scale=1.1)

    def test_fixed_constants(self):
        self.assertEqual(ALPHA, 1.6732632423543772)
        self.assertEqual(SCALE, 1.0507009873554805)

    def test_activation_matches_reference(self):
        for z in (-50.0, -2.0, -0.5, -1e-12, 0.0, -0.0, 1e-12, 0.5, 2.0,
                  50.0):
            self.assertTrue(close(SELUSequence._activate(z), selu(z),
                                  atol=1e-15))
            out = SELUSequence._activate(z)
            self.assertTrue(
                close(SELUSequence._activation_derivative(out),
                      selu_deriv(z), atol=1e-14),
                "z=%r: %r vs %r"
                % (z, SELUSequence._activation_derivative(out),
                   selu_deriv(z)))

    def test_zero_derivative_uses_unique_negative_branch_convention(self):
        # At z == 0 the output is exactly 0.0 and the derivative is the
        # z <= 0 branch value scale * alpha * exp(0) = scale * alpha, which
        # differs from the positive-branch slope scale.
        self.assertEqual(SELUSequence._activate(0.0), 0.0)
        self.assertEqual(SELUSequence._activate(-0.0), 0.0)
        self.assertEqual(SELUSequence._activation_derivative(0.0),
                         SCALE_ALPHA)
        self.assertNotEqual(SCALE_ALPHA, SCALE)


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

    def test_extreme_negative_preactivations_stay_finite_and_saturate(self):
        # A hugely negative z saturates the exponential to exactly
        # -scale * alpha with no overflow possible (exp is only ever
        # evaluated at z < 0).
        seq = SELUSequence(Linear([1.0, 0.0], 0.0))
        got = seq.forward([[-1e6], [1e6], [-1e6]])
        self.assertEqual(got, [-SCALE_ALPHA, SCALE * 1e6, -SCALE_ALPHA])
        self.assertTrue(all(math.isfinite(v) for v in got))
        self.assertEqual(seq.step([-1e6]), -SCALE_ALPHA)
        self.assertEqual(seq.step([1e6]), SCALE * 1e6)

    def test_extreme_finite_negative_inputs_through_recurrence(self):
        # Large finite inputs combined with finite parameters stay finite on
        # the negative branch in batch, stream and stepwise entries.
        rows = [[-1e100], [-1e150], [-1e200]]
        seq = SELUSequence(Linear([1.0, 0.5], 0.0))
        got = seq.forward(rows, truncate=2)
        self.assertTrue(all(math.isfinite(v) for v in got))
        self.assertEqual(got[0], -SCALE_ALPHA)
        stream = SELUSequence(Linear([1.0, 0.5], 0.0))
        stream.start_stream(truncate=2)
        streamed = [stream.step(row) for row in rows]
        self.assertEqual(streamed, got)
        stream.finish_stream()

    def test_zero_preactivation_outputs_exact_zero(self):
        # z == 0 exactly: the output is exactly 0.0, positive sign.
        seq = SELUSequence(Linear([1.0, 0.0], -1.0))
        got = seq.forward([[1.0]])
        self.assertEqual(got, [0.0])
        self.assertEqual(math.copysign(1.0, got[0]), 1.0)

    def test_negative_preactivation_uses_scaled_expm1(self):
        seq = SELUSequence(Linear([1.0, 0.0], 0.0))
        self.assertEqual(seq.forward([[-2.0]]),
                         [SCALE_ALPHA * math.expm1(-2.0)])
        self.assertEqual(seq.forward([[-1e-12]])[0],
                         SCALE_ALPHA * math.expm1(-1e-12))

    def test_outputs_are_bounded_below(self):
        # SELU maps every finite z into (-scale * alpha, +inf).
        seq = fresh()
        got = seq.forward(ROWS, truncate=2, initial_hidden=-0.9)
        self.assertTrue(all(v > -SCALE_ALPHA - 1e-14 for v in got))

    def test_positive_branch_overflow_rejected_atomically(self):
        # z finite but scale * z past the double maximum: ValueError, no
        # infinity committed anywhere.
        self.assertFalse(math.isfinite(SCALE * OVERFLOWING_POSITIVE_Z))

        def probe(make_seq, call):
            seq = make_seq()
            expect_value_error(call(seq))
            self.assertTrue(math.isfinite(seq.hidden))
            self.assertTrue(all(math.isfinite(v) for v in seq.outputs))
            if seq.linear.last is not None:
                self.assertTrue(all(math.isfinite(v)
                                    for v in seq.linear.last))
            return seq

        # Plain stepwise entry.
        seq = probe(lambda: SELUSequence(Linear([1.0, 0.0], 0.0)),
                    lambda s: lambda: s.step([OVERFLOWING_POSITIVE_Z]))
        self.assertEqual(seq.outputs, [])
        # Batch entry (failure at the second row).
        seq = probe(lambda: SELUSequence(Linear([1.0, 0.0], 0.0)),
                    lambda s: lambda: s.forward(
                        [[0.5], [OVERFLOWING_POSITIVE_Z]]))
        self.assertEqual(seq.outputs, [])
        # Stream entry.
        seq = SELUSequence(Linear([1.0, 0.0], 0.0))
        seq.start_stream()
        expect_value_error(lambda: seq.step([OVERFLOWING_POSITIVE_Z]))
        self.assertEqual(len(seq._stream["outputs"]), 0)
        self.assertTrue(math.isfinite(seq.hidden))
        # The failed session is still usable and finishes normally.
        seq.step([0.5])
        self.assertEqual(seq.finish_stream(), [selu(0.5)])


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
        if seeds is not None:
            # Explicit seeds override the standard boundary constants;
            # frozen_boundaries already supplies the carry/reset value for
            # every boundary past 0, and position 0 follows h0 directly
            # (no seed is given there).
            frozen = frozen_boundaries(rows, truncate, carry, weight, bias,
                                       h0, start_set)
            frozen.update({i: seed for i, seed in enumerate(seeds)
                           if seed is not None})
        else:
            frozen = frozen_boundaries(rows, truncate, carry, weight, bias,
                                       h0, start_set)
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
        # Gradient with respect to the forward's initial_hidden.
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

    def test_grads_match_finite_differences_truncated(self):
        for carry in (False, True):
            self._check_pass(W, B, ROWS, 0.2, 2, carry, 0.1)

    def test_grads_match_finite_differences_with_segment_starts(self):
        starts = [False, True, False, True, False]
        self._check_pass(W, B, ROWS, -0.1, 3, True, 0.05, starts)

    def test_grads_match_finite_differences_with_explicit_seeds(self):
        # Position 0 keeps h0 (no seed there); an explicit detached seed is
        # given at the truncate boundary 2 and boundary 4 resets to zero.
        seeds = [None, None, -0.5, None, None]
        self._check_pass(W, B, ROWS, 0.3, 2, False, 0.2, seeds=seeds)

    def test_grads_match_finite_differences_multi_feature(self):
        weight, bias = [0.4, -0.3, 0.2], 0.1
        rows = [[0.5, 0.1], [-0.2, 0.7], [0.9, -0.4]]
        self._check_pass(weight, bias, rows, 0.0, 2, True, 0.2)

    def test_backward_returns_only_input_grads(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        got = seq.backward(GO)
        self.assertIsInstance(got, list)
        self.assertTrue(all(isinstance(v, float) for v in got))
        # Same accumulated parameter gradients as the boundaries variant.
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
        # Matches one backward with doubled upstream on a fresh instance.
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
        # No parameter gradient accumulates for an empty pass.
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)

    def test_local_derivative_is_scale_on_positive_preactivation(self):
        # One step, unit input weight, z > 0: d_pre = go * scale.
        seq = SELUSequence(Linear([1.0, 0.0], 0.0))
        out = seq.forward([[1.0]])[0]  # z = 1 -> out = scale
        self.assertEqual(out, SCALE)
        seq.backward([0.8])
        self.assertTrue(close(seq.linear.grad[0], 0.8 * SCALE, atol=1e-15))
        self.assertTrue(close(seq.linear.grad_bias, 0.8 * SCALE, atol=1e-15))

    def test_local_derivative_at_and_below_zero(self):
        # At z == 0 the published derivative is scale * alpha; for z < 0 it
        # is scale * alpha * exp(z), recovered from the cached output as
        # scale * alpha + output.
        for z in (-1.0, -0.25, 0.0):
            seq = SELUSequence(Linear([1.0, 0.0], 0.0))
            out = seq.forward([[z]])[0]
            self.assertEqual(out, selu(z))
            got = seq.backward([0.8])
            want = 0.8 * SCALE_ALPHA * math.exp(z)
            self.assertTrue(close(got[0], want, atol=1e-13),
                            "z=%r: %r vs %r" % (z, got[0], want))
            self.assertTrue(close(seq.linear.grad[0],
                                  0.8 * SCALE_ALPHA * math.exp(z) * z,
                                  atol=1e-13))
            self.assertTrue(close(seq.linear.grad[1], 0.0, atol=1e-15))
            self.assertTrue(close(seq.linear.grad_bias,
                                  0.8 * SCALE_ALPHA * math.exp(z),
                                  atol=1e-13))

    def test_saturated_negative_derivative_is_zero(self):
        # z hugely negative: output is exactly -scale * alpha and the
        # derivative scale * alpha + output is exactly 0.0.
        seq = SELUSequence(Linear([1.0, 0.0], 0.0))
        self.assertEqual(seq.forward([[-1e6]])[0], -SCALE_ALPHA)
        got = seq.backward([0.8])
        self.assertEqual(got, [0.0])
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)

    def test_backward_uses_cached_trajectory_after_parameter_update(self):
        # apply_gradients() after the forward must not change the gradients
        # of the recorded pass: the pass replays its cached weights, and the
        # result stays deterministic across repeated backwards.
        seq = fresh()
        seq.forward(ROWS, truncate=2, initial_hidden=0.2)
        ref = fresh()
        ref.forward(ROWS, truncate=2, initial_hidden=0.2)
        ref.backward_with_boundaries(GO, 0.1)

        # Updating the parameters between forward and backward changes none
        # of the recorded pass's input gradients (the cached forward-time
        # weights are replayed).
        seq.linear.apply_gradients(0.37)
        input_grads, grad_init, boundary_grads = \
            seq.backward_with_boundaries(GO, 0.1)
        self.assertTrue(allclose(input_grads,
                                 ref.backward_with_boundaries(GO, 0.1)[0],
                                 atol=1e-14))
        # The recorded pass's gradient increment accumulates onto the
        # post-update Linear exactly as it would on a zero-grad replay.
        replay = fresh()
        replay.forward(ROWS, truncate=2, initial_hidden=0.2)
        replay.backward_with_boundaries(GO, 0.1)
        self.assertTrue(allclose(seq.linear.grad, replay.linear.grad,
                                 atol=1e-13))
        self.assertTrue(close(seq.linear.grad_bias, replay.linear.grad_bias,
                              atol=1e-13))
        # Repeating the same backward on the same cache stays deterministic.
        again_inputs, again_init, again_bounds = \
            seq.backward_with_boundaries(GO, 0.1)
        self.assertTrue(allclose(again_inputs, input_grads, atol=1e-14))
        self.assertEqual(again_init, grad_init)
        self.assertEqual(again_bounds, boundary_grads)

    def test_overflowing_backward_accumulates_nothing(self):
        # A huge finite upstream overflows the recurrence; the atomic
        # finiteness check rejects the whole pass and accumulates nothing.
        seq = SELUSequence(Linear([1.0, 0.5], 0.0))
        seq.forward(ROWS)
        before = (list(seq.linear.grad), seq.linear.grad_bias)
        expect_value_error(
            lambda: seq.backward_with_initial_hidden([1e308] * len(ROWS),
                                                     1e308))
        self.assertEqual((list(seq.linear.grad), seq.linear.grad_bias),
                         before)
        # A finite retry succeeds and yields finite gradients.
        got = seq.backward([0.1] * len(ROWS))
        self.assertTrue(all(math.isfinite(v) for v in got))


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
            self.assertTrue(allclose(got, batch.outputs, atol=1e-14))
            self.assertEqual(stream.finish_stream(), stream.outputs)
            self.assertTrue(allclose(stream.outputs, batch.outputs,
                                     atol=1e-14))
            gb = batch.backward_with_boundaries(GO, 0.1)
            gs = stream.backward_with_boundaries(GO, 0.1)
            self.assertTrue(allclose(gb[0], gs[0], atol=1e-14))
            self.assertTrue(close(gb[1], gs[1], atol=1e-14))
            self.assertEqual([i for i, _ in gb[2]], [i for i, _ in gs[2]])
            self.assertTrue(allclose([g for _, g in gb[2]],
                                     [g for _, g in gs[2]], atol=1e-14))
            self.assertTrue(allclose(batch.linear.grad, stream.linear.grad,
                                     atol=1e-14))

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
        self.assertTrue(allclose(stream.outputs, batch.outputs, atol=1e-14))
        self.assertTrue(allclose(stream.backward(GO), batch.backward(GO),
                                 atol=1e-14))

    def test_stream_segment_hiddens_match_batch_value_by_value(self):
        seeds = [0.4, None, -0.6, None, None]
        batch = fresh()
        got = batch.forward(ROWS, truncate=2, carry_hidden=True,
                            segment_hiddens=seeds)
        stream = fresh()
        stream.start_stream(initial_hidden=0.0, truncate=2,
                            carry_hidden=True)
        stepped = [stream.step(list(row), segment_hidden=seed)
                   for row, seed in zip(ROWS, seeds)]
        self.assertEqual(stepped, got)
        self.assertEqual(stream.finish_stream(), got)
        gb = batch.backward_with_boundaries(GO, 0.12)
        gs = stream.backward_with_boundaries(GO, 0.12)
        self.assertTrue(allclose(gb[0], gs[0], atol=1e-14))
        self.assertEqual(gb[2], gs[2])

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
        # After finishing, backward works.
        seq.backward([0.1])


class CheckpointRestoreTest(unittest.TestCase):
    def test_checkpoint_restore_round_trip(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        seq.backward(GO)
        cp = seq.checkpoint()
        snap = seq.export_state()
        # Diverge, then restore: everything returns to the snapshot.
        seq.step([9.9])
        seq.linear.apply_gradients(0.5)
        seq.start_stream()
        seq.step([0.1])
        self.assertIsNone(seq.restore(cp))
        self.assertEqual(seq.export_state(), snap)
        # Restoring is repeatable and keeps backward deterministic.
        seq.step([0.4])
        seq.restore(cp)
        self.assertEqual(seq.export_state(), snap)
        a = seq.backward(GO)
        seq.restore(cp)
        b = seq.backward(GO)
        self.assertEqual(a, b)

    def test_checkpoint_is_instance_bound(self):
        src = fresh()
        src.forward(ROWS)
        cp = src.checkpoint()
        other = fresh()
        expect_value_error(lambda: other.restore(cp))
        # Instances of the other sequence classes reject it too (and it
        # rejects theirs).
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
        # The restored session continues exactly as the original would.
        twin = fresh()
        twin.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        for row in ROWS[:3]:
            twin.step(list(row))
        for row in ROWS[3:]:
            self.assertEqual(seq.step(list(row)), twin.step(list(row)))
        self.assertEqual(seq.finish_stream(), twin.finish_stream())

    def test_corrupt_checkpoint_rejected_atomically(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        seq.backward(GO)

        def snap():
            return seq.export_state()

        cp = seq.checkpoint()
        cp._hidden = NAN
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap())
        cp = seq.checkpoint()
        cp._fwd["outputs"][0] = INF
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap())
        cp = seq.checkpoint()
        cp._linear_weight = [0.1]  # width mismatch
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap())
        # An inconsistent trajectory (prev_hidden breaking the truncation
        # relation) is rejected as well.
        cp = seq.checkpoint()
        cp._fwd["prev_hiddens"][2] = 12.5
        expect_value_error(lambda: seq.restore(cp))
        self.assertEqual(seq.export_state(), snap())


class ExportImportTest(unittest.TestCase):
    def test_kind_and_version(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        state = seq.export_state()
        self.assertEqual(state["kind"], "SELUSequenceState")
        self.assertEqual(state["version"], 2)
        for cls in OTHER_CLASSES:
            self.assertNotEqual(state["kind"],
                                cls(Linear(list(W), B)).export_state()["kind"])
        # JSON round trip preserves the value exactly.
        self.assertEqual(json.loads(json.dumps(state)), state)

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

    def test_version_1_state_imports_with_none_seeds(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        state = json.loads(json.dumps(src.export_state()))
        state["version"] = 1
        del state["forward"]["segment_hiddens"]
        dst = fresh()
        dst.import_state(state)
        self.assertEqual(dst.export_state(), src.export_state())
        self.assertEqual(dst.backward(GO), src.backward(GO))
        # A version-1 open-stream record migrates the same way.
        streaming = fresh()
        streaming.start_stream(initial_hidden=0.25, truncate=2)
        streaming.step(ROWS[0])
        streaming.step(ROWS[1])
        stream_state = json.loads(json.dumps(streaming.export_state()))
        stream_state["version"] = 1
        del stream_state["stream"]["segment_hiddens"]
        dst2 = fresh()
        dst2.import_state(stream_state)
        self.assertEqual(dst2.export_state(), streaming.export_state())

    def test_repeated_import_is_stable(self):
        src = fresh()
        src.forward(ROWS, truncate=2, initial_hidden=0.2)
        state = src.export_state()
        dst = fresh()
        dst.import_state(state)
        first = dst.export_state()
        dst.forward([[9.0]])
        dst.import_state(copy.deepcopy(state))
        self.assertEqual(dst.export_state(), first)
        dst.step([0.1])
        dst.import_state(json.loads(json.dumps(state)))
        self.assertEqual(dst.export_state(), first)

    def test_cross_kind_states_are_rejected(self):
        selu_seq = fresh()
        selu_seq.forward(ROWS, truncate=2)
        others = [cls(Linear(list(W), B)) for cls in OTHER_CLASSES]
        for other in others:
            other.forward(ROWS, truncate=2)
        selu_snap = selu_seq.export_state()
        other_snaps = [other.export_state() for other in others]
        other_states = [other.export_state() for other in others]
        for other, other_state in zip(others, other_states):
            expect_value_error(
                lambda other_state=other_state:
                selu_seq.import_state(other_state))
            expect_value_error(
                lambda other=other: other.import_state(selu_snap))
        # Rejections leave every instance untouched.
        self.assertEqual(selu_seq.export_state(), selu_snap)
        for other, other_snap in zip(others, other_snaps):
            self.assertEqual(other.export_state(), other_snap)

    def test_width_mismatch_rejected(self):
        src = fresh([0.4, -0.3, 0.2], 0.1)
        src.forward([[0.5, 0.1], [-0.2, 0.7]])
        state = src.export_state()
        dst = fresh()
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

        def reject(tamper):
            state = copy.deepcopy(src.export_state())
            tamper(state)
            expect_value_error(lambda state=state: dst.import_state(state))
            self.assertEqual(dst.export_state(), snap)

        reject(lambda s: s.update(version=3))
        reject(lambda s: s.update(width=3))
        reject(lambda s: s.update(hidden=NAN))
        reject(lambda s: s["forward"].update(truncate=0))
        reject(lambda s: s["linear"].update(last=[1.0]))
        reject(lambda s: s.update(kind="ELUSequenceState"))
        reject(lambda s: s.update(kind="SELUSequenceState-typo"))
        reject(lambda s: s["forward"]["outputs"].__setitem__(0, NAN))
        # Trajectory inconsistency: a prev_hidden the boundary rules never
        # produce.
        reject(lambda s: s["forward"]["prev_hiddens"].__setitem__(2, 3.5))
        # A stream record and a batch record present at the same time.
        reject(lambda s: s.update(stream=copy.deepcopy(s["forward"])))
        # Structural corruption: missing field / wrong types.
        reject(lambda s: s.pop("hidden"))
        reject(lambda s: s["forward"].update(boundaries="not-a-list"))
        # Importing a non-dict is rejected too.
        expect_value_error(lambda: dst.import_state([]))
        expect_value_error(lambda: dst.import_state("nope"))
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
        # Bad rows: wrong width, non-numbers, bools, non-finite values.
        for bad_row in ([0.1, 0.2], [], [True], ["0.1"], [NAN], [INF], [BIG]):
            expect_value_error(
                lambda bad_row=bad_row: fresh().forward([bad_row]))
        expect_value_error(lambda: fresh().forward([[0.1], [NAN]]))

    def test_segment_hiddens_validation(self):
        seq = fresh()
        # Length mismatch.
        expect_value_error(lambda: seq.forward(
            ROWS, segment_hiddens=[None, None]))
        # A number at a non-boundary position.
        expect_value_error(lambda: seq.forward(
            ROWS, truncate=2,
            segment_hiddens=[None, 0.5, None, None, None]))
        # Non-number / non-finite entries.
        for bad in (True, "0.1", NAN, INF, BIG):
            expect_value_error(lambda bad=bad: seq.forward(
                ROWS, truncate=2,
                segment_hiddens=[bad, None, None, None, None]))
        # A value outside a stream is a RuntimeError via step.
        with self.assertRaises(RuntimeError):
            fresh().step([0.1], segment_hidden=0.5)

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
        # Rejected calls accumulate nothing.
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
        # Position 0 is a boundary: a finite seed there is allowed.
        self.assertTrue(math.isfinite(seq.step([0.1], segment_hidden=0.5)))
        # A seed at the following non-boundary step is rejected.
        expect_value_error(lambda: seq.step([0.1], segment_hidden=0.5))
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

    def test_overflowing_forward_leaves_everything_untouched(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        # A finite, legal row whose linear arithmetic overflows.
        seq.linear.weight = [1e308, 0.5]
        before = self.snapshot(seq)
        expect_value_error(lambda: seq.forward([[10.0]]))
        self.assertEqual(self.snapshot(seq), before)
        # The earlier cache is still back-propagatable.
        seq.backward(GO)

    def test_positive_branch_overflow_step_leaves_everything_untouched(self):
        # SELU-specific: the linear pre-activation is finite, only the
        # scaled activation overflows. The base non-stream step path has no
        # activation failure point for the other classes, so the SELU
        # override must roll the Linear forward record back itself.
        seq = SELUSequence(Linear([1.0, 0.5], 0.0))
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        before = self.snapshot(seq)
        expect_value_error(
            lambda: seq.step([OVERFLOWING_POSITIVE_Z]))
        self.assertEqual(self.snapshot(seq), before)
        # The old batch cache still back-propagates exactly as before.
        ref = SELUSequence(Linear([1.0, 0.5], 0.0))
        ref.forward(ROWS, truncate=2)
        ref.backward(GO)
        self.assertEqual(seq.backward(GO), ref.backward(GO))
        # A legal retry continues the trajectory normally.
        self.assertTrue(math.isfinite(seq.step([0.3])))

    def test_overflowing_stream_step_leaves_session_untouched(self):
        seq = SELUSequence(Linear([0.4, -0.3], 0.15))
        seq.start_stream(initial_hidden=0.1, truncate=2)
        seq.step([0.8])
        seq.step([-0.5])
        seq.linear.weight = [1e308, 0.5]
        before = self.snapshot(seq)
        expect_value_error(lambda: seq.step([10.0], segment_start=True))
        self.assertEqual(self.snapshot(seq), before)
        # A legal retry succeeds and the session finishes normally.
        seq.linear.weight = [0.4, -0.3]
        seq.step([10.0], segment_start=True)
        self.assertEqual(len(seq.finish_stream()), 3)

    def test_positive_branch_overflow_stream_step_leaves_session_untouched(self):
        seq = SELUSequence(Linear([1.0, 0.5], 0.0))
        seq.start_stream(initial_hidden=0.1, truncate=2)
        seq.step([0.8])
        before = self.snapshot(seq)
        expect_value_error(
            lambda: seq.step([OVERFLOWING_POSITIVE_Z]))
        self.assertEqual(self.snapshot(seq), before)
        # Session continues exactly as if the failed call never happened.
        self.assertEqual(seq.step([-0.5]),
                         selu(1.0 * -0.5 + 0.5 * selu(0.8 + 0.5 * 0.1)
                              + 0.0))
        self.assertEqual(len(seq.finish_stream()), 2)

    def test_overflowing_backward_accumulates_nothing(self):
        seq = fresh()
        seq.forward(ROWS)
        seq.backward(GO)
        before = self.snapshot(seq)
        # 1e308 + 1e308 overflows to inf inside the recurrence; the atomic
        # finiteness check rejects the whole pass.
        expect_value_error(
            lambda: seq.backward_with_initial_hidden([1e308] * len(ROWS),
                                                     1e308))
        self.assertEqual(self.snapshot(seq), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
