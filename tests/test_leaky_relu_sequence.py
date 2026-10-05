"""LeakyReLUSequence: the leaky rectified-linear sequence unit.

Mirrors the TanhSequence/SigmoidSequence/ReLUSequence conventions:
independent reference formulas, finite-difference gradient checks,
stream/batch equivalence, checkpoint and state-migration rules (with a kind
tag distinct from the other sequence classes), validation and failure
atomicity. The negative slope is fixed at 0.01; the constructor accepts
only a Linear.
"""
import copy
import json
import math
import unittest

import app
from app import (LeakyReLUSequence, Linear, ReLUSequence, SigmoidSequence,
                 SoftplusSequence, TanhSequence)

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400  # finite int, but overflows double arithmetic

EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-7

SLOPE = 0.01

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

def leaky(z):
    if z > 0:
        return z
    if z == 0:
        return 0.0
    return SLOPE * z


def sequence_outputs(weight, bias, rows, h0, truncate, carry, starts=(),
                     frozen=None):
    """leaky(w_in.row + w_h*h_prev + b), with the published boundary rules.

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
        hidden = leaky(z)
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
    return LeakyReLUSequence(Linear(list(weight), bias))


class SmokeTest(unittest.TestCase):
    def test_module_exposes_leaky_relu_sequence(self):
        self.assertTrue(hasattr(app, "LeakyReLUSequence"))
        self.assertTrue(issubclass(LeakyReLUSequence, TanhSequence))

    def test_construction_rules_match_tanh(self):
        expect_value_error(lambda: LeakyReLUSequence("not a linear"))
        expect_value_error(lambda: LeakyReLUSequence(Linear([1.0])))
        seq = LeakyReLUSequence(Linear([0.4, -0.3], 0.15))
        self.assertEqual(seq.d, 1)
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])

    def test_no_slope_configuration(self):
        # The constructor takes only the Linear; there is no extra
        # configurable slope argument.
        with self.assertRaises(TypeError):
            LeakyReLUSequence(Linear([0.4, -0.3], 0.15), 0.2)
        with self.assertRaises(TypeError):
            LeakyReLUSequence(Linear([0.4, -0.3], 0.15), negative_slope=0.2)


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

    def test_forward_with_segment_starts_merges_boundaries(self):
        starts = [False, True, False, False, True]
        for carry in (False, True):
            seq = fresh()
            got = seq.forward(ROWS, truncate=3, carry_hidden=carry,
                              initial_hidden=0.2, segment_starts=starts)
            want = sequence_outputs(W, B, ROWS, 0.2, 3, carry,
                                    starts={1, 4})
            self.assertTrue(allclose(got, want, atol=1e-15))

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

    def test_extreme_preactivations_stay_finite_and_exact(self):
        # |z| around 1e12: a huge positive z passes through as exactly z, a
        # huge negative z is scaled to exactly 0.01 * z; no overflow is
        # possible.
        seq = LeakyReLUSequence(Linear([1e6, 0.0], 0.0))
        got = seq.forward([[1e6], [-1e6], [1e6]])
        self.assertEqual(got, [1e12, 0.01 * -1e12, 1e12])
        self.assertTrue(all(math.isfinite(v) for v in got))
        self.assertEqual(seq.step([-1e6]), 0.01 * -1e12)
        self.assertEqual(seq.step([1e6]), 1e12)

    def test_zero_preactivation_outputs_exact_zero(self):
        # z == 0 exactly: the output is exactly 0.0, not a signed product.
        seq = LeakyReLUSequence(Linear([1.0, 0.0], -1.0))
        got = seq.forward([[1.0]])
        self.assertEqual(got, [0.0])
        self.assertEqual(math.copysign(1.0, got[0]), 1.0)

    def test_negative_preactivation_is_scaled_by_fixed_slope(self):
        seq = LeakyReLUSequence(Linear([1.0, 0.0], 0.0))
        self.assertEqual(seq.forward([[-2.0]]), [0.01 * -2.0])


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
        # Matches one backward with doubled upstream on a fresh instance.
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
        # No parameter gradient accumulates for an empty pass.
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)

    def test_local_derivative_is_one_on_positive_output(self):
        # One step, unit input weight, z > 0: d_pre = go * 1.
        seq = LeakyReLUSequence(Linear([1.0, 0.0], 0.0))
        out = seq.forward([[1.0]])[0]  # z = 1 -> out = 1
        self.assertEqual(out, 1.0)
        seq.backward([0.8])
        self.assertTrue(close(seq.linear.grad[0], 0.8, atol=1e-15))
        self.assertTrue(close(seq.linear.grad_bias, 0.8, atol=1e-15))

    def test_local_derivative_is_slope_on_non_positive_output(self):
        # One step, unit input weight, z <= 0: d_pre = go * 0.01.
        for z in (-1.0, 0.0):
            seq = LeakyReLUSequence(Linear([1.0, 0.0], 0.0))
            out = seq.forward([[z]])[0]
            self.assertEqual(out, leaky(z))
            got = seq.backward([0.8])
            self.assertTrue(close(got[0], 0.8 * SLOPE, atol=1e-15))
            self.assertTrue(close(seq.linear.grad[0], 0.8 * SLOPE * z,
                                  atol=1e-15))
            self.assertTrue(close(seq.linear.grad[1], 0.0, atol=1e-15))
            self.assertTrue(close(seq.linear.grad_bias, 0.8 * SLOPE,
                                  atol=1e-15))


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
        # Restoring is repeatable.
        seq.step([0.4])
        seq.restore(cp)
        self.assertEqual(seq.export_state(), snap)

    def test_checkpoint_is_instance_bound(self):
        src = fresh()
        src.forward(ROWS)
        cp = src.checkpoint()
        other = fresh()
        expect_value_error(lambda: other.restore(cp))
        # Instances of the other sequence classes reject it too (and it
        # rejects theirs).
        for cls in (TanhSequence, SigmoidSequence, SoftplusSequence,
                    ReLUSequence):
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


class ExportImportTest(unittest.TestCase):
    def test_kind_and_version(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        state = seq.export_state()
        self.assertEqual(state["kind"], "LeakyReLUSequenceState")
        self.assertEqual(state["version"], 2)
        for cls in (TanhSequence, SigmoidSequence, SoftplusSequence,
                    ReLUSequence):
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

    def test_cross_kind_states_are_rejected(self):
        leaky_seq = fresh()
        leaky_seq.forward(ROWS, truncate=2)
        others = [TanhSequence(Linear(list(W), B)),
                  SigmoidSequence(Linear(list(W), B)),
                  SoftplusSequence(Linear(list(W), B)),
                  ReLUSequence(Linear(list(W), B))]
        for other in others:
            other.forward(ROWS, truncate=2)
        leaky_snap = leaky_seq.export_state()
        other_snaps = [other.export_state() for other in others]
        other_states = [other.export_state() for other in others]
        for other, other_state in zip(others, other_states):
            expect_value_error(
                lambda other_state=other_state:
                leaky_seq.import_state(other_state))
            expect_value_error(
                lambda other=other: other.import_state(leaky_snap))
        # Rejections leave every instance untouched.
        self.assertEqual(leaky_seq.export_state(), leaky_snap)
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
                lambda s: s.update(kind="ReLUSequenceState"),
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
        # Bad rows: wrong width, non-numbers, bools, non-finite values.
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

    def test_overflowing_stream_step_leaves_session_untouched(self):
        seq = LeakyReLUSequence(Linear([0.4, -0.3], 0.15))
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
