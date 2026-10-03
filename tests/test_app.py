import ast
import math
import subprocess
import sys
import unittest
from pathlib import Path

import app
from app import Linear, TanhSequence

REPO_ROOT = Path(__file__).resolve().parent.parent

# Central finite-difference step and the comparison tolerances. The analytic
# quantities are tanh expressions (third derivatives bounded), so h=1e-6
# leaves truncation error around 1e-13..1e-10; the tolerances below are loose
# by comparison while still being tight enough to catch any broken recurrence.
EPS = 1e-6
ATOL = 1e-7
RTOL = 1e-7


def close(a, b, atol=ATOL, rtol=RTOL):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def allclose(xs, ys, atol=ATOL, rtol=RTOL):
    return len(xs) == len(ys) and all(
        close(a, b, atol, rtol) for a, b in zip(xs, ys)
    )


def central(f, eps=EPS):
    return (f(eps) - f(-eps)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# Independent reference formulas. These deliberately re-derive the public
# math instead of calling the implementation, so a test compares code against
# the published recurrence rather than code against code.
# ---------------------------------------------------------------------------

def linear_output(weight, bias, x):
    return sum(w * a for w, a in zip(weight, x)) + bias


def sequence_outputs(weight, bias, xs, h0, truncate, carry, frozen=None):
    """tanh(w_in*x + w_h*h_prev + b), with published segment-boundary rules.

    ``frozen`` maps a segment-start index to a constant hidden value; it plays
    the role of the detached boundary constant used by truncated backprop.
    """
    outs = []
    hidden = h0
    for i, x in enumerate(xs):
        if truncate is not None and i % truncate == 0:
            if i == 0:
                hidden = h0
            elif frozen is not None:
                hidden = frozen[i]
            elif carry:
                hidden = outs[-1]
            else:
                hidden = 0.0
        hidden = math.tanh(weight[0] * x + weight[1] * hidden + bias)
        outs.append(hidden)
    return outs


def frozen_boundaries(xs, truncate, carry, weight, bias, h0):
    """Boundary constants taken from the unperturbed trajectory."""
    if truncate is None:
        return {}
    base = sequence_outputs(weight, bias, xs, h0, truncate, carry)
    frozen = {}
    for i in range(1, len(xs)):
        if i % truncate == 0:
            frozen[i] = base[i - 1] if carry else 0.0
    return frozen


def objective(weight, bias, xs, h0, grad_outputs, grad_hidden,
              truncate, carry, frozen):
    """Scalar loss whose derivatives are exactly the published gradients.

    L = sum_t go_t * out_t + grad_hidden * out_{-1}; with no steps the
    terminal hidden state *is* the initial hidden state, so L = grad_hidden*h0.
    """
    outs = sequence_outputs(weight, bias, xs, h0, truncate, carry, frozen)
    if not outs:
        return grad_hidden * h0
    return sum(g * o for g, o in zip(grad_outputs, outs)) + grad_hidden * outs[-1]


def fresh_sequence(weight, bias):
    return TanhSequence(Linear(list(weight), bias))


class SmokeTest(unittest.TestCase):
    def test_module_exposes_public_entries(self):
        self.assertTrue(hasattr(app, "Linear"))
        self.assertTrue(hasattr(app, "TanhSequence"))

    def test_demo_output_matches_reference(self):
        # Locks the shipped example: demo.py prints truncate=2 over [1,2,3]
        # with weight [0.4, 0.2]; the second segment starts from zero.
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "demo.py")],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        printed = ast.literal_eval(proc.stdout.strip())
        o1 = math.tanh(0.8 + 0.2 * math.tanh(0.4))
        # Third element starts a fresh segment from zero (no carry).
        expected = [math.tanh(0.4), o1, math.tanh(1.2)]
        self.assertTrue(allclose(printed, expected, atol=1e-12))


class LinearForwardTest(unittest.TestCase):
    def test_forward_is_weighted_sum_plus_bias(self):
        weight, bias, x = [0.3, -0.7, 0.5], 0.2, [0.4, -1.1, 0.9]
        lin = Linear(weight, bias)
        self.assertTrue(close(lin.forward(x), linear_output(weight, bias, x),
                              atol=1e-15))

    def test_default_bias_is_zero_and_ints_accepted(self):
        lin = Linear([1, 2])
        self.assertEqual(lin.bias, 0.0)
        self.assertEqual(lin.forward([3, 4]), 11)

    def test_forward_accepts_any_weight_width(self):
        lin = Linear([0.25], -0.5)
        self.assertTrue(close(lin.forward([4.0]), 0.5))

    def test_construction_rejects_non_numbers_and_bools(self):
        for bad_weight in ([1, True], [1, "2"], [1.0, None], [1, object()]):
            with self.assertRaises(ValueError):
                Linear(bad_weight)
        with self.assertRaises(ValueError):
            Linear([1.0], bias=False)
        with self.assertRaises(ValueError):
            Linear([1.0], bias="0.0")

    def test_forward_validates_length_and_types(self):
        lin = Linear([1.0, 2.0])
        with self.assertRaises(ValueError):
            lin.forward([1.0])
        with self.assertRaises(ValueError):
            lin.forward([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            lin.forward([True, 2.0])
        with self.assertRaises(ValueError):
            lin.forward([1.0, "2"])
        with self.assertRaises(ValueError):
            lin.forward(b"ab")

    def test_rejected_forward_caches_nothing(self):
        lin = Linear([1.0, 2.0])
        with self.assertRaises(ValueError):
            lin.forward([1.0])
        self.assertIsNone(lin.last)
        with self.assertRaises(RuntimeError):
            lin.backward(1.0)


class LinearBackwardTest(unittest.TestCase):
    WEIGHT = [0.3, -0.7, 0.5]
    BIAS = 0.2
    X = [0.4, -1.1, 0.9]
    G = 0.6

    def setUp(self):
        self.lin = Linear(list(self.WEIGHT), self.BIAS)
        self.lin.forward(self.X)

    def test_backward_formula(self):
        input_grads = self.lin.backward(self.G)
        self.assertTrue(allclose(input_grads, [self.G * w for w in self.WEIGHT],
                                 atol=1e-15))
        self.assertTrue(allclose(self.lin.grad, [self.G * a for a in self.X],
                                 atol=1e-15))
        self.assertTrue(close(self.lin.grad_bias, self.G, atol=1e-15))

    def test_grads_match_central_finite_differences(self):
        analytic = self.lin.backward(self.G)
        w, b, x, g = self.WEIGHT, self.BIAS, self.X, self.G
        for t in range(len(x)):
            fd = central(lambda d, t=t: linear_output(
                w, b, [a + (d if k == t else 0.0) for k, a in enumerate(x)]) * g)
            self.assertTrue(close(analytic[t], fd),
                            "input grad %d: %r vs %r" % (t, analytic[t], fd))
        for t in range(len(w)):
            fd = central(lambda d, t=t: linear_output(
                [wi + (d if k == t else 0.0) for k, wi in enumerate(w)], b, x) * g)
            self.assertTrue(close(self.lin.grad[t], fd),
                            "weight grad %d: %r vs %r" % (t, self.lin.grad[t], fd))
        fd_bias = central(lambda d: linear_output(w, b + d, x) * g)
        self.assertTrue(close(self.lin.grad_bias, fd_bias))

    def test_input_grads_use_weights_as_of_forward_time(self):
        old_weight = list(self.WEIGHT)
        self.lin.weight = [9.9, -9.9, 9.9]
        input_grads = self.lin.backward(self.G)
        self.assertTrue(allclose(input_grads, [self.G * w for w in old_weight]))
        # Parameter grads still pair the cached inputs with this upstream.
        self.assertTrue(allclose(self.lin.grad, [self.G * a for a in self.X]))

    def test_repeated_backward_accumulates(self):
        first = self.lin.backward(self.G)
        second = self.lin.backward(0.25)
        # Each call returns a fresh list describing only that call's upstream.
        self.assertIsNot(first, second)
        self.assertTrue(allclose(second, [0.25 * w for w in self.WEIGHT]))
        total_g = self.G + 0.25
        self.assertTrue(allclose(self.lin.grad, [total_g * a for a in self.X]))
        self.assertTrue(close(self.lin.grad_bias, total_g))
        # A single backward with the combined upstream must give identical
        # accumulated parameter gradients.
        ref = Linear(list(self.WEIGHT), self.BIAS)
        ref.forward(self.X)
        ref.backward(total_g)
        self.assertTrue(allclose(self.lin.grad, ref.grad))
        self.assertTrue(close(self.lin.grad_bias, ref.grad_bias))

    def test_zero_grad_clears_only_gradients(self):
        self.lin.backward(self.G)
        weight_before, bias_before = list(self.lin.weight), self.lin.bias
        cache_before = self.lin.last
        self.lin.zero_grad()
        self.assertEqual(self.lin.grad, [0.0, 0.0, 0.0])
        self.assertEqual(self.lin.grad_bias, 0.0)
        self.assertEqual(self.lin.weight, weight_before)
        self.assertEqual(self.lin.bias, bias_before)
        self.assertIs(self.lin.last, cache_before)
        # The cache is still usable after zeroing.
        again = self.lin.backward(self.G)
        self.assertTrue(allclose(again, [self.G * w for w in weight_before]))

    def test_apply_gradients_uses_published_update(self):
        grads = [0.01, -0.02, 0.03]
        self.lin.zero_grad()
        for t, value in enumerate(grads):
            self.lin.grad[t] = value
        self.lin.grad_bias = -0.04
        old_weight, old_bias, lr = list(self.lin.weight), self.lin.bias, 0.1
        self.lin.apply_gradients(lr)
        self.assertTrue(allclose(
            self.lin.weight,
            [w - lr * g for w, g in zip(old_weight, grads)], atol=1e-16))
        self.assertTrue(close(self.lin.bias, old_bias - lr * -0.04, atol=1e-16))
        # apply_gradients does not zero the gradients itself.
        self.assertTrue(allclose(self.lin.grad, grads))
        # A forward with the updated parameters follows the updated formula.
        self.assertTrue(close(
            self.lin.forward(self.X), linear_output(self.lin.weight, self.lin.bias, self.X)))

    def test_apply_gradients_rejects_bad_learning_rate(self):
        weight_before, bias_before = list(self.lin.weight), self.lin.bias
        for bad_lr in (True, "0.1", None):
            with self.assertRaises(ValueError):
                self.lin.apply_gradients(bad_lr)
        self.assertEqual(self.lin.weight, weight_before)
        self.assertEqual(self.lin.bias, bias_before)

    def test_backward_requires_successful_forward(self):
        with self.assertRaises(RuntimeError):
            Linear([1.0]).backward(1.0)

    def test_backward_validates_upstream_and_changes_nothing_on_failure(self):
        self.lin.backward(self.G)
        grads_before, bias_grad_before = list(self.lin.grad), self.lin.grad_bias
        cache_before = list(self.lin.last)
        for bad in (True, "1.0", None):
            with self.assertRaises(ValueError):
                self.lin.backward(bad)
        self.assertEqual(self.lin.last, cache_before)
        self.assertEqual(self.lin.grad, grads_before)
        self.assertEqual(self.lin.grad_bias, bias_grad_before)


# Small fixed sample reused across the sequence tests.
W = [0.4, -0.3]
B = 0.15
XS = [0.8, -0.5, 1.2, -0.7, 0.3]
ROWS = [[x] for x in XS]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]
GH = 0.35
H0 = 0.25


class StepBatchEquivalenceTest(unittest.TestCase):
    def assert_step_matches_batch(self, h0):
        stepped = fresh_sequence(W, B)
        # The public entry for supplying an initial hidden state to a stepwise
        # walk is an empty batch forward that commits h0.
        stepped.forward([], initial_hidden=h0)
        step_outs = [stepped.step([x]) for x in XS]

        batched = fresh_sequence(W, B)
        batch_outs = batched.forward(ROWS, initial_hidden=h0)

        self.assertTrue(allclose(step_outs, batch_outs, atol=1e-15))
        self.assertTrue(close(stepped.hidden, batched.hidden, atol=1e-15))
        self.assertTrue(allclose(
            step_outs, sequence_outputs(W, B, XS, h0, None, False), atol=1e-15))

    def test_step_matches_batch_from_zero(self):
        self.assert_step_matches_batch(0.0)

    def test_step_matches_batch_with_initial_hidden(self):
        self.assert_step_matches_batch(H0)

    def test_step_appends_output_and_updates_hidden(self):
        seq = fresh_sequence(W, B)
        out = seq.step([XS[0]])
        expected = math.tanh(W[0] * XS[0] + B)
        self.assertTrue(close(out, expected, atol=1e-15))
        self.assertTrue(close(seq.hidden, expected, atol=1e-15))
        self.assertEqual(seq.outputs, [out])

    def test_batch_forward_starts_at_zero_ignoring_prior_steps(self):
        # Without initial_hidden a batch walk always starts at 0.0, even if
        # step() had advanced the state beforehand.
        seq = fresh_sequence(W, B)
        seq.step([5.0])
        outs = seq.forward(ROWS[:2])
        self.assertTrue(allclose(
            outs, sequence_outputs(W, B, XS[:2], 0.0, None, False), atol=1e-15))

    def test_step_invalidates_batch_cache_but_bad_step_changes_nothing(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        # A rejected step leaves hidden, outputs, the Linear record and the
        # batch cache exactly as they were.
        hidden_before, outputs_before = seq.hidden, list(seq.outputs)
        last_before = list(seq.linear.last)
        grad_before = list(seq.linear.grad)
        for bad_row in ([], [1.0, 2.0], ["x"], [True], 7, "ab"):
            with self.assertRaises(ValueError):
                seq.step(bad_row)
        self.assertEqual(seq.hidden, hidden_before)
        self.assertEqual(seq.outputs, outputs_before)
        self.assertEqual(seq.linear.last, last_before)
        self.assertEqual(seq.linear.grad, grad_before)
        cached = seq.backward(GO)
        self.assertEqual(len(cached), len(ROWS))
        # A successful step mixes the walk with the recorded batch pass, so
        # that cache can no longer be back-propagated.
        seq.step([0.2])
        with self.assertRaises(RuntimeError):
            seq.backward(GO)


class SequenceForwardTest(unittest.TestCase):
    def _check_trajectory(self, xs, truncate, carry, h0):
        seq = fresh_sequence(W, B)
        outs = seq.forward([[x] for x in xs], truncate, carry, h0)
        expected = sequence_outputs(W, B, xs, h0, truncate, carry)
        self.assertTrue(allclose(outs, expected, atol=1e-12))
        if xs:
            self.assertTrue(close(seq.hidden, expected[-1], atol=1e-12))
        else:
            # An empty walk commits the starting hidden state with no outputs.
            self.assertTrue(close(seq.hidden, h0, atol=1e-15))
        self.assertEqual(seq.outputs, outs)
        return outs

    def test_no_truncation_trajectory(self):
        self._check_trajectory(XS, None, False, 0.0)
        self._check_trajectory(XS, None, False, H0)

    def test_truncate_one_severs_every_link(self):
        reset_outs = self._check_trajectory(XS, 1, False, 0.0)
        carry_outs = self._check_trajectory(XS, 1, True, H0)
        # With truncate=1 every step is its own segment: reset starts from
        # zero, carry starts from the previous output.
        self.assertTrue(allclose(
            reset_outs, [math.tanh(W[0] * x + B) for x in XS], atol=1e-15))
        self.assertTrue(close(carry_outs[0], math.tanh(W[0] * XS[0] + W[1] * H0 + B),
                              atol=1e-15))

    def test_truncation_in_the_middle_and_past_the_end(self):
        for truncate in (2, 3):
            self._check_trajectory(XS, truncate, False, 0.0)
            self._check_trajectory(XS, truncate, True, H0)
        # truncate larger than the sequence is a single segment.
        self._check_trajectory(XS, 100, False, H0)
        self._check_trajectory(XS, len(XS), True, 0.0)
        # A sequence ending exactly on a boundary.
        self._check_trajectory(XS[:4], 2, False, 0.0)
        self._check_trajectory(XS[:4], 2, True, H0)

    def test_empty_and_single_step_sequences(self):
        for truncate, carry in ((None, False), (1, False), (1, True), (2, True)):
            self._check_trajectory([], truncate, carry, H0)
            self._check_trajectory(XS[:1], truncate, carry, H0)

    def test_carried_boundary_starts_from_numeric_copy_of_prior_end(self):
        # Even with a nonzero initial state, a carried new segment begins at
        # the previous segment's final hidden value, and a reset one at zero.
        outs = self._check_trajectory(XS, 2, True, H0)
        self.assertTrue(close(outs[2], math.tanh(W[0] * XS[2] + W[1] * outs[1] + B),
                              atol=1e-15))
        outs_reset = self._check_trajectory(XS, 2, False, H0)
        self.assertTrue(close(outs_reset[2], math.tanh(W[0] * XS[2] + B),
                              atol=1e-15))
        # The nonzero h0 must not leak past the first reset boundary.
        self.assertFalse(close(outs_reset[2],
                               math.tanh(W[0] * XS[2] + W[1] * outs_reset[1] + B)))

    def test_invalid_arguments_raise_value_error(self):
        seq = fresh_sequence(W, B)
        for bad_truncate in (0, -2, 1.5, True, "2"):
            with self.assertRaises(ValueError):
                seq.forward(ROWS, truncate=bad_truncate)
        for bad_carry in (1, 0, "true", None):
            with self.assertRaises(ValueError):
                seq.forward(ROWS, carry_hidden=bad_carry)
        for bad_hidden in (True, "0.0", 1j):
            with self.assertRaises(ValueError):
                seq.forward(ROWS, initial_hidden=bad_hidden)

    def test_invalid_arguments_change_no_state(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        seq.backward(GO)
        snapshot = (seq.hidden, list(seq.outputs), list(seq.linear.last),
                    list(seq.linear.weight), list(seq.linear.grad),
                    seq.linear.grad_bias)
        for kwargs in ({"truncate": 0}, {"truncate": -2}, {"truncate": 1.5},
                       {"truncate": True}, {"carry_hidden": 1},
                       {"carry_hidden": 0}, {"initial_hidden": "x"},
                       {"initial_hidden": True}):
            with self.assertRaises(ValueError):
                seq.forward(ROWS, **kwargs)
            self.assertEqual((seq.hidden, list(seq.outputs),
                              list(seq.linear.last), list(seq.linear.weight),
                              list(seq.linear.grad), seq.linear.grad_bias),
                             snapshot)
        # The pre-error cache is still back-propagatable.
        self.assertEqual(len(seq.backward(GO)), len(ROWS))

    def test_bad_rows_mid_traversal_rolls_back_all_state(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        snapshot = (
            seq.hidden,
            list(seq.outputs),
            list(seq.linear.last),
            list(seq.linear.grad),
            seq.linear.grad_bias,
        )
        for bad_rows in ([[1.0], [1.0, 2.0]], [[1.0], ["x"]],
                         [[True]], "ab", [[1.0], None]):
            with self.assertRaises(ValueError):
                seq.forward(bad_rows, truncate=2)
            self.assertEqual(seq.hidden, snapshot[0])
            self.assertEqual(seq.outputs, snapshot[1])
            self.assertEqual(seq.linear.last, snapshot[2])
            self.assertEqual(seq.linear.grad, snapshot[3])
            self.assertEqual(seq.linear.grad_bias, snapshot[4])
        # The pre-existing cache is still fully usable: this second backward
        # returns that call's input grads (one GO) while parameter grads are
        # now the sum of two GO passes.
        again = seq.backward(GO)
        once = fresh_sequence(W, B)
        once.forward(ROWS, truncate=2)
        expected_inputs = once.backward(GO)
        doubled = fresh_sequence(W, B)
        doubled.forward(ROWS, truncate=2)
        doubled.backward([2 * g for g in GO])
        self.assertTrue(allclose(again, expected_inputs))
        self.assertTrue(allclose(seq.linear.grad, doubled.linear.grad))
        self.assertTrue(close(seq.linear.grad_bias, doubled.linear.grad_bias))

    def test_failed_first_forward_leaves_no_half_pass(self):
        seq = fresh_sequence(W, B)
        with self.assertRaises(ValueError):
            seq.forward([[1.0], [1.0, 2.0]], truncate=2)
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])
        self.assertIsNone(seq.linear.last)
        # No successful forward means no cache to back-propagate.
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        with self.assertRaises(RuntimeError):
            seq.backward_with_initial_hidden(GO, GH)


class SequenceBackwardTest(unittest.TestCase):
    def _run_gradient_case(self, xs, truncate, carry, h0, grad_hidden):
        """Check every published gradient of one forward against finite diffs.

        Segment-boundary hidden values are held at their unperturbed values
        during finite differencing, mirroring the published "detached
        constant" semantics; boundary agreement therefore proves that no
        derivative crosses a segment boundary.
        """
        go = GO[:len(xs)]
        frozen = frozen_boundaries(xs, truncate, carry, W, B, h0)

        seq = fresh_sequence(W, B)
        seq.forward([[x] for x in xs], truncate, carry, h0)
        if grad_hidden:
            input_grads, grad_initial = seq.backward_with_initial_hidden(go, grad_hidden)
        else:
            input_grads = seq.backward(go)
            # Seed zero must reproduce backward() and additionally expose the
            # initial-hidden gradient through the seed-aware entry point.
            twin = fresh_sequence(W, B)
            twin.forward([[x] for x in xs], truncate, carry, h0)
            twin_inputs, grad_initial = twin.backward_with_initial_hidden(go, 0.0)
            self.assertTrue(allclose(twin_inputs, input_grads, atol=1e-15))

        def obj(weight, bias, inputs, initial):
            return objective(weight, bias, inputs, initial, go, grad_hidden,
                             truncate, carry, frozen)

        for t in range(len(xs)):
            fd = central(lambda d, t=t: obj(
                W, B, [a + (d if k == t else 0.0) for k, a in enumerate(xs)], h0))
            self.assertTrue(close(input_grads[t], fd),
                            "input grad t=%d: analytic %r fd %r" % (t, input_grads[t], fd))
        fd_w0 = central(lambda d: obj([W[0] + d, W[1]], B, xs, h0))
        fd_w1 = central(lambda d: obj([W[0], W[1] + d], B, xs, h0))
        fd_b = central(lambda d: obj(W, B + d, xs, h0))
        fd_h0 = central(lambda d: obj(W, B, xs, h0 + d))
        self.assertTrue(close(seq.linear.grad[0], fd_w0),
                        "w_in grad: %r vs %r" % (seq.linear.grad[0], fd_w0))
        self.assertTrue(close(seq.linear.grad[1], fd_w1),
                        "w_h grad: %r vs %r" % (seq.linear.grad[1], fd_w1))
        self.assertTrue(close(seq.linear.grad_bias, fd_b),
                        "bias grad: %r vs %r" % (seq.linear.grad_bias, fd_b))
        self.assertTrue(close(grad_initial, fd_h0),
                        "initial hidden grad: %r vs %r" % (grad_initial, fd_h0))
        return seq, input_grads

    def test_no_truncation_gradients(self):
        self._run_gradient_case(XS, None, False, 0.0, 0.0)
        self._run_gradient_case(XS, None, False, H0, GH)

    def test_truncate_one_gradients(self):
        self._run_gradient_case(XS, 1, False, 0.0, 0.0)
        self._run_gradient_case(XS, 1, True, H0, GH)

    def test_truncation_middle_gradients(self):
        self._run_gradient_case(XS, 2, False, 0.0, 0.0)
        self._run_gradient_case(XS, 2, True, H0, GH)
        self._run_gradient_case(XS, 3, False, H0, 0.0)
        self._run_gradient_case(XS, 3, True, 0.0, GH)

    def test_truncation_longer_than_sequence_gradients(self):
        self._run_gradient_case(XS, 100, False, 0.0, 0.0)
        self._run_gradient_case(XS, 100, True, H0, GH)

    def test_empty_sequence_gradients(self):
        seq = fresh_sequence(W, B)
        seq.forward([], truncate=2, carry_hidden=True, initial_hidden=H0)
        input_grads, grad_initial = seq.backward_with_initial_hidden([], GH)
        self.assertEqual(input_grads, [])
        # No steps: the terminal hidden state is the initial hidden state.
        self.assertTrue(close(grad_initial, GH, atol=1e-15))
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)
        plain = seq.backward([])
        self.assertEqual(plain, [])

    def test_single_step_gradients(self):
        self._run_gradient_case(XS[:1], None, False, H0, GH)
        self._run_gradient_case(XS[:1], 1, True, H0, GH)

    def test_cut_boundary_makes_a_numerical_difference(self):
        # Guard against a vacuous severing test: with carry=True the true
        # (graph-connected) derivative through the boundary must differ from
        # the published truncated one.
        frozen = frozen_boundaries(XS, 2, True, W, B, 0.0)
        cut = central(lambda d: objective(
            [W[0], W[1] + d], B, XS, 0.0, GO, 0.0, 2, True, frozen))
        full = central(lambda d: objective(
            [W[0], W[1] + d], B, XS, 0.0, GO, 0.0, 2, True, None))
        self.assertFalse(close(cut, full, atol=1e-9))

    def test_segment_start_outputs_and_local_grads_checked(self):
        # Explicit per-segment bookkeeping for truncate=2, carry=True:
        # segment A = steps 0..1 from h0, segment B = steps 2..3 starting from
        # a detached copy of outs[1]. Both segment-start outputs, their local
        # parameter contributions and the within-segment chain terms are
        # pinned against a hand-derived formula.
        seq = fresh_sequence(W, B)
        outs = seq.forward(ROWS[:4], truncate=2, carry_hidden=True, initial_hidden=H0)
        self.assertTrue(close(outs[0], math.tanh(W[0] * XS[0] + W[1] * H0 + B),
                              atol=1e-15))
        self.assertTrue(close(outs[2], math.tanh(W[0] * XS[2] + W[1] * outs[1] + B),
                              atol=1e-15))
        go = GO[:4]
        input_grads = seq.backward(go)

        d1 = go[1] * (1.0 - outs[1] ** 2)
        d3 = go[3] * (1.0 - outs[3] ** 2)
        d0 = (go[0] + d1 * W[1]) * (1.0 - outs[0] ** 2)
        d2 = (go[2] + d3 * W[1]) * (1.0 - outs[2] ** 2)
        dpre = [d0, d1, d2, d3]
        self.assertTrue(allclose(input_grads, [d * W[0] for d in dpre], atol=1e-14))
        # Weight grad pairs every step with its local inputs; the second
        # segment starts from the detached constant outs[1], not from h0.
        self.assertTrue(close(
            seq.linear.grad[0],
            sum(d * x for d, x in zip(dpre, XS[:4])), atol=1e-14))
        self.assertTrue(close(
            seq.linear.grad[1],
            d0 * H0 + d1 * outs[0] + d2 * outs[1] + d3 * outs[2], atol=1e-14))
        self.assertTrue(close(seq.linear.grad_bias, sum(dpre), atol=1e-14))

        # Direct severing proof: perturbing the final-step upstream gradient
        # changes steps 3 and 2 (same segment) but never steps 1 or 0.
        shifted = fresh_sequence(W, B)
        shifted.forward(ROWS[:4], truncate=2, carry_hidden=True, initial_hidden=H0)
        shifted_grads = shifted.backward([go[0], go[1], go[2], go[3] + 0.7])
        self.assertTrue(close(shifted_grads[0], input_grads[0], atol=1e-15))
        self.assertTrue(close(shifted_grads[1], input_grads[1], atol=1e-15))
        self.assertFalse(close(shifted_grads[2], input_grads[2], atol=1e-9))
        self.assertFalse(close(shifted_grads[3], input_grads[3], atol=1e-9))

    def test_repeated_backward_accumulates_not_overwrites(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        first = seq.backward(GO)
        go2 = [0.15 * (t + 1) for t in range(len(XS))]
        second = seq.backward(go2)

        ref = fresh_sequence(W, B)
        ref.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        combined = [a + b for a, b in zip(GO, go2)]
        combined_inputs = ref.backward(combined)
        first_again = _inputs_only(W, B, ROWS, 2, True, H0, GO)
        second_again = _inputs_only(W, B, ROWS, 2, True, H0, go2)
        self.assertTrue(allclose(first, first_again))
        self.assertTrue(allclose(second, second_again))
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias))
        self.assertTrue(allclose(combined_inputs, [a + b for a, b in zip(first, second)]))

    def test_seed_entry_accumulates_like_plain_backward(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, initial_hidden=H0)
        plain_inputs = seq.backward(GO)
        seeded_inputs, grad_initial = seq.backward_with_initial_hidden(GO, 0.0)
        self.assertTrue(allclose(seeded_inputs, plain_inputs, atol=1e-15))
        # Zero terminal seed adds nothing at the last step.
        ref = fresh_sequence(W, B)
        ref.forward(ROWS, initial_hidden=H0)
        ref.backward(GO)
        self.assertTrue(allclose(seq.linear.grad, [2 * g for g in ref.linear.grad]))

    def test_zero_grad_clears_grads_but_keeps_cache_and_parameters(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        weight_before, bias_before = list(seq.linear.weight), seq.linear.bias
        seq.linear.zero_grad()
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)
        self.assertEqual(seq.linear.weight, weight_before)
        self.assertEqual(seq.linear.bias, bias_before)
        # The sequence cache survives zeroing and backprops identically.
        again = seq.backward(GO)
        ref = fresh_sequence(W, B)
        ref.forward(ROWS, truncate=2)
        self.assertTrue(allclose(again, ref.backward(GO)))

    def test_parameter_update_uses_lr_formula_and_cached_weights_frozen(self):
        seq = fresh_sequence(W, B)
        outs = seq.forward(ROWS, initial_hidden=H0)
        input_grads, _ = seq.backward_with_initial_hidden(GO, GH)
        grads_before = list(seq.linear.grad)
        lr = 0.1
        old_weight, old_bias = list(seq.linear.weight), seq.linear.bias
        seq.linear.apply_gradients(lr)
        self.assertTrue(allclose(
            seq.linear.weight,
            [w - lr * g for w, g in zip(old_weight, grads_before)], atol=1e-16))
        self.assertTrue(close(seq.linear.bias,
                              old_bias - lr * seq.linear.grad_bias, atol=1e-16))
        # The recorded pass keeps its forward-time weights: input grads of a
        # second backward on the same cache are unchanged, and parameter grad
        # contributions are computed from cached outputs.
        again, _ = seq.backward_with_initial_hidden(GO, GH)
        self.assertTrue(allclose(again, input_grads, atol=1e-15))
        # Forward outputs after the update follow the updated parameters.
        moved = seq.forward(ROWS, initial_hidden=H0)
        self.assertTrue(allclose(
            moved, sequence_outputs(seq.linear.weight, seq.linear.bias, XS, H0, None, False)))
        self.assertFalse(allclose(moved, outs, atol=1e-9))

    def test_bad_grad_lists_raise_and_leave_no_partial_accumulation(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        snapshot = (list(seq.linear.grad), seq.linear.grad_bias)
        for bad in ([0.0] * 4, [0.0] * 6, [], [1.0, "x", 0.0, 0.0, 0.0],
                    [True, 0.0, 0.0, 0.0, 0.0], "00000", object()):
            with self.assertRaises(ValueError):
                seq.backward(bad)
            with self.assertRaises(ValueError):
                seq.backward_with_initial_hidden(bad, GH)
        self.assertEqual(seq.linear.grad, snapshot[0])
        self.assertEqual(seq.linear.grad_bias, snapshot[1])
        # Bad terminal seed is rejected before accumulation as well.
        for bad_seed in (True, "0.0", None):
            with self.assertRaises(ValueError):
                seq.backward_with_initial_hidden(GO, bad_seed)
        self.assertEqual(seq.linear.grad, snapshot[0])
        self.assertEqual(seq.linear.grad_bias, snapshot[1])
        # Cache still fully usable after every rejected call.
        self.assertEqual(len(seq.backward(GO)), len(ROWS))

    def test_backward_requires_live_cache(self):
        fresh = fresh_sequence(W, B)
        with self.assertRaises(RuntimeError):
            fresh.backward(GO)
        with self.assertRaises(RuntimeError):
            fresh.backward_with_initial_hidden(GO, GH)

    def test_reset_invalidates_cache_but_preserves_linear_state(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2, initial_hidden=H0)
        seq.backward(GO)
        weight, grads = list(seq.linear.weight), list(seq.linear.grad)
        grad_bias, last = seq.linear.grad_bias, list(seq.linear.last)
        self.assertNotEqual(seq.outputs, [])

        seq.reset()
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])
        # Cache invalidation is observable through the public entry points.
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        with self.assertRaises(RuntimeError):
            seq.backward_with_initial_hidden(GO, GH)
        # The wrapped layer's parameters, accumulated gradients and its own
        # last-forward record are untouched by a sequence reset.
        self.assertEqual(seq.linear.weight, weight)
        self.assertEqual(seq.linear.grad, grads)
        self.assertEqual(seq.linear.grad_bias, grad_bias)
        self.assertEqual(seq.linear.last, last)
        # A rejected backward after reset changes nothing observable: the
        # following forward/backward works normally from the clean state.
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        seq.forward(ROWS, truncate=2, initial_hidden=H0)
        self.assertEqual(len(seq.backward(GO)), len(ROWS))


def _inputs_only(weight, bias, rows, truncate, carry, h0, go):
    """Analytic input-grad list of one fresh pass, used for accumulation."""
    seq = fresh_sequence(weight, bias)
    seq.forward(rows, truncate, carry, h0)
    return seq.backward(go)


def _outputs_with_boundary_values(weight, bias, xs, h0, boundary_set, values):
    """Scalar reference traversal whose segment starts are exactly
    ``boundary_set``; every start other than 0 reads a frozen constant from
    ``values`` (detached, exactly as the published boundary semantics)."""
    outs = []
    hidden = h0
    for i, x in enumerate(xs):
        if i in boundary_set:
            hidden = h0 if i == 0 else values[i]
        hidden = math.tanh(weight[0] * x + weight[1] * hidden + bias)
        outs.append(hidden)
    return outs


class BoundaryGradientTest(unittest.TestCase):
    def _boundary_values(self, outs, boundary_set, carry):
        return {i: (outs[i - 1] if carry else 0.0)
                for i in boundary_set if i > 0}

    def _check_boundary_finite_differences(self, xs, truncate, carry, h0,
                                           go, gh, declared=None):
        """Every reported boundary gradient must equal the central finite
        difference of the loss with respect to that start's frozen hidden
        constant, with all other boundary constants held fixed."""
        seq = fresh_sequence(W, B)
        outs = seq.forward([[x] for x in xs], truncate, carry, h0,
                           segment_starts=declared)
        input_grads, grad_initial, boundary_grads = \
            seq.backward_with_boundaries(go, gh)

        if truncate is not None:
            boundary_set = {i for i in range(len(xs)) if i % truncate == 0}
        else:
            boundary_set = {0}
        if declared is not None:
            boundary_set |= {i for i, flag in enumerate(declared) if flag}
        expected_indices = sorted(i for i in boundary_set if i > 0)
        self.assertEqual([i for i, _ in boundary_grads], expected_indices)

        values = self._boundary_values(outs, boundary_set, carry)

        def loss(perturbed):
            traj = _outputs_with_boundary_values(
                W, B, xs, h0, boundary_set, perturbed)
            return (sum(g * o for g, o in zip(go, traj)) + gh * traj[-1]
                    if traj else gh * h0)

        for index, gradient in boundary_grads:
            fd = central(lambda d, index=index: loss(
                {**values, index: values[index] + d}))
            self.assertTrue(close(gradient, fd),
                            "boundary grad i=%d: analytic %r fd %r"
                            % (index, gradient, fd))
        return seq, input_grads, grad_initial, boundary_grads

    def test_first_two_returns_match_seeded_entry(self):
        for truncate, carry, h0, gh in (
                (None, False, 0.0, 0.0), (2, False, 0.0, GH),
                (2, True, H0, GH), (3, True, 0.0, GH), (1, True, H0, GH)):
            seeded = fresh_sequence(W, B)
            seeded.forward(ROWS, truncate, carry, h0)
            ref_inputs, ref_initial = seeded.backward_with_initial_hidden(GO, gh)
            bounded = fresh_sequence(W, B)
            bounded.forward(ROWS, truncate, carry, h0)
            input_grads, grad_initial, _ = \
                bounded.backward_with_boundaries(GO, gh)
            self.assertTrue(allclose(input_grads, ref_inputs, atol=1e-15))
            self.assertTrue(close(grad_initial, ref_initial, atol=1e-15))

    def test_boundary_gradients_match_finite_differences(self):
        self._check_boundary_finite_differences(XS, 2, True, H0, GO, GH)
        self._check_boundary_finite_differences(XS, 2, False, 0.0, GO, GH)
        self._check_boundary_finite_differences(XS, 3, True, 0.0, GO, 0.0)
        self._check_boundary_finite_differences(XS, 1, True, H0, GO, GH)
        self._check_boundary_finite_differences(XS, 1, False, H0, GO, GH)
        self._check_boundary_finite_differences(XS[:4], 2, True, H0, GO[:4], 0.0)

    def test_truncate_and_segment_starts_merge_and_dedup(self):
        # truncate=2 yields starts 0,2,4; declarations add 3 and repeat 2.
        declared = [False, False, True, True, False]
        _, _, _, boundary_grads = self._check_boundary_finite_differences(
            XS, 2, True, H0, GO, GH, declared=declared)
        self.assertEqual([i for i, _ in boundary_grads], [2, 3, 4])

    def test_explicit_segment_starts_without_truncate(self):
        declared = [False, True, False, True, False]
        _, _, _, boundary_grads = self._check_boundary_finite_differences(
            XS, None, False, H0, GO, GH, declared=declared)
        self.assertEqual([i for i, _ in boundary_grads], [1, 3])
        # With carry the constants are the prior segment ends.
        _, _, _, carried = self._check_boundary_finite_differences(
            XS, None, True, H0, GO, GH, declared=declared)
        self.assertEqual([i for i, _ in carried], [1, 3])

    def test_no_nonzero_index_boundaries_returns_empty_list(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, initial_hidden=H0)
        _, _, boundary_grads = seq.backward_with_boundaries(GO, GH)
        self.assertEqual(boundary_grads, [])
        # truncate longer than the sequence only has the index-zero start.
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=100, carry_hidden=True, initial_hidden=H0)
        _, _, boundary_grads = seq.backward_with_boundaries(GO, GH)
        self.assertEqual(boundary_grads, [])
        # Single step: only start 0 exists.
        seq = fresh_sequence(W, B)
        seq.forward(ROWS[:1], truncate=1, carry_hidden=True, initial_hidden=H0)
        _, _, boundary_grads = seq.backward_with_boundaries(GO[:1], GH)
        self.assertEqual(boundary_grads, [])

    def test_empty_sequence_returns_empty_inputs_passthrough_seed(self):
        seq = fresh_sequence(W, B)
        seq.forward([], truncate=2, carry_hidden=True, initial_hidden=H0)
        input_grads, grad_initial, boundary_grads = \
            seq.backward_with_boundaries([], GH)
        self.assertEqual(input_grads, [])
        self.assertTrue(close(grad_initial, GH, atol=1e-15))
        self.assertEqual(boundary_grads, [])
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.linear.grad_bias, 0.0)

    def test_stream_cache_reports_same_boundaries(self):
        streamed = fresh_sequence(W, B)
        streamed.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        for x in XS:
            streamed.step([x])
        streamed.finish_stream()
        s_inputs, s_initial, s_boundaries = \
            streamed.backward_with_boundaries(GO, GH)

        batched = fresh_sequence(W, B)
        batched.forward(ROWS, 2, True, H0)
        b_inputs, b_initial, b_boundaries = \
            batched.backward_with_boundaries(GO, GH)
        self.assertTrue(allclose(s_inputs, b_inputs, atol=1e-15))
        self.assertTrue(close(s_initial, b_initial, atol=1e-15))
        self.assertEqual(len(s_boundaries), len(b_boundaries))
        for (si, sg), (bi, bg) in zip(s_boundaries, b_boundaries):
            self.assertEqual(si, bi)
            self.assertTrue(close(sg, bg, atol=1e-15))

    def test_repeated_calls_accumulate_like_merged_upstream(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        _, _, first_boundaries = seq.backward_with_boundaries(GO, GH)
        go2 = [0.11 * (t + 1) for t in range(len(XS))]
        gh2 = -0.4
        _, _, second_boundaries = seq.backward_with_boundaries(go2, gh2)

        merged = fresh_sequence(W, B)
        merged.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        _, _, merged_boundaries = merged.backward_with_boundaries(
            [a + b for a, b in zip(GO, go2)], GH + gh2)
        self.assertTrue(allclose(seq.linear.grad, merged.linear.grad))
        self.assertTrue(close(seq.linear.grad_bias, merged.linear.grad_bias))

        totals = {}
        for boundaries in (first_boundaries, second_boundaries):
            for index, gradient in boundaries:
                totals[index] = totals.get(index, 0.0) + gradient
        self.assertEqual(sorted(totals),
                         [i for i, _ in merged_boundaries])
        for index, gradient in merged_boundaries:
            self.assertTrue(close(totals[index], gradient))

    def test_multi_feature_boundary_gradients(self):
        seq = TanhSequence(Linear(list(WM), BM))
        outs = seq.forward(ROWSM, truncate=2, carry_hidden=True,
                           initial_hidden=H0)
        input_grads, grad_initial, boundary_grads = \
            seq.backward_with_boundaries(GOM, GH)
        self.assertEqual([i for i, _ in boundary_grads], [2, 4])
        for row_grads in input_grads:
            self.assertIsInstance(row_grads, list)
            self.assertEqual(len(row_grads), 2)

        d = len(WM) - 1

        def loss(values):
            traj = []
            hidden = H0
            for i, row in enumerate(ROWSM):
                if i % 2 == 0:
                    hidden = H0 if i == 0 else values[i]
                pre = sum(WM[k] * row[k] for k in range(d)) + WM[d] * hidden + BM
                hidden = math.tanh(pre)
                traj.append(hidden)
            return sum(g * o for g, o in zip(GOM, traj)) + GH * traj[-1]

        values = {2: outs[1], 4: outs[3]}
        for index, gradient in boundary_grads:
            fd = central(lambda dd, index=index: loss(
                {**values, index: values[index] + dd}))
            self.assertTrue(close(gradient, fd))

        twin = TanhSequence(Linear(list(WM), BM))
        twin.forward(ROWSM, 2, True, H0)
        ref_inputs, ref_initial = twin.backward_with_initial_hidden(GOM, GH)
        for a, b in zip(input_grads, ref_inputs):
            self.assertTrue(allclose(a, b, atol=1e-15))
        self.assertTrue(close(grad_initial, ref_initial, atol=1e-15))

    def test_validation_and_state_rules_match_seeded_entry(self):
        seq = fresh_sequence(W, B)
        with self.assertRaises(RuntimeError):
            seq.backward_with_boundaries(GO)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        snapshot = (list(seq.linear.grad), seq.linear.grad_bias)
        for bad in ([0.0] * 4, [0.0] * 6, [], [1.0, "x", 0.0, 0.0, 0.0],
                    [True, 0.0, 0.0, 0.0, 0.0], "00000", object()):
            with self.assertRaises(ValueError):
                seq.backward_with_boundaries(bad, GH)
        for bad_seed in (True, "0.0", None):
            with self.assertRaises(ValueError):
                seq.backward_with_boundaries(GO, bad_seed)
        self.assertEqual(seq.linear.grad, snapshot[0])
        self.assertEqual(seq.linear.grad_bias, snapshot[1])
        # Cache still fully usable after every rejected call.
        self.assertEqual(
            len(seq.backward_with_boundaries(GO)[0]), len(ROWS))

    def test_open_session_raises_runtime_error(self):
        seq = fresh_sequence(W, B)
        seq.start_stream(truncate=2)
        seq.step([XS[0]])
        with self.assertRaises(RuntimeError):
            seq.backward_with_boundaries([0.0])
        # The session remains open and the finished cache works normally.
        self.assertEqual(len(seq.finish_stream()), 1)
        _, _, boundary_grads = seq.backward_with_boundaries([0.0])
        self.assertEqual(boundary_grads, [])


class StreamSessionTest(unittest.TestCase):
    def _run_stream(self, xs, truncate=None, carry=False, h0=None):
        seq = fresh_sequence(W, B)
        seq.start_stream(initial_hidden=h0, truncate=truncate, carry_hidden=carry)
        stepped = [seq.step([x]) for x in xs]
        returned = seq.finish_stream()
        return seq, stepped, returned

    def test_stream_outputs_match_batch_trajectory(self):
        for truncate, carry in ((None, False), (1, False), (1, True),
                                (2, False), (2, True), (3, True), (100, False)):
            for h0 in (None, H0):
                seq, stepped, returned = self._run_stream(
                    XS, truncate, carry, h0)
                expected = sequence_outputs(W, B, XS, h0 or 0.0, truncate, carry)
                self.assertTrue(allclose(stepped, expected, atol=1e-15),
                                "truncate=%r carry=%r h0=%r" % (truncate, carry, h0))
                # finish_stream returns the hidden states in arrival order.
                self.assertTrue(allclose(returned, expected, atol=1e-15))
                self.assertEqual(seq.outputs, returned)
                self.assertIsNot(returned, seq.outputs)
                self.assertTrue(close(seq.hidden, expected[-1], atol=1e-15))

    def test_start_stream_returns_none_and_clears_outputs(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, initial_hidden=H0)
        self.assertNotEqual(seq.outputs, [])
        self.assertIsNone(seq.start_stream(initial_hidden=H0))
        self.assertEqual(seq.outputs, [])
        self.assertTrue(close(seq.hidden, H0, atol=1e-15))

    def test_start_stream_default_initial_hidden_is_zero(self):
        seq = fresh_sequence(W, B)
        seq.start_stream()
        self.assertEqual(seq.hidden, 0.0)
        out = seq.step([XS[0]])
        self.assertTrue(close(out, math.tanh(W[0] * XS[0] + B), atol=1e-15))

    def test_start_stream_invalidates_cache_but_keeps_linear_state(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        weight, grads = list(seq.linear.weight), list(seq.linear.grad)
        grad_bias, last = seq.linear.grad_bias, list(seq.linear.last)
        seq.start_stream()
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        self.assertEqual(seq.linear.weight, weight)
        self.assertEqual(seq.linear.grad, grads)
        self.assertEqual(seq.linear.grad_bias, grad_bias)
        self.assertEqual(seq.linear.last, last)

    def _check_stream_gradients(self, xs, truncate, carry, h0, grad_hidden):
        """Stream gradients must equal the batch gradients, which the batch
        tests already pinned against finite differences of the reference."""
        go = GO[:len(xs)]
        seq, _, _ = self._run_stream(xs, truncate, carry, h0)
        input_grads, grad_initial = seq.backward_with_initial_hidden(go, grad_hidden)

        ref = fresh_sequence(W, B)
        ref.forward([[x] for x in xs], truncate, carry, h0)
        ref_inputs, ref_initial = ref.backward_with_initial_hidden(go, grad_hidden)
        self.assertTrue(allclose(input_grads, ref_inputs, atol=1e-15))
        self.assertTrue(close(grad_initial, ref_initial, atol=1e-15))
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad, atol=1e-15))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias, atol=1e-15))

        # Independent finite-difference check of the stream gradients, with
        # boundary hidden values frozen at their unperturbed trajectory.
        frozen = frozen_boundaries(xs, truncate, carry, W, B, h0)
        obj = lambda weight, bias, inputs, initial: objective(
            weight, bias, inputs, initial, go, grad_hidden, truncate, carry, frozen)
        for t in range(len(xs)):
            fd = central(lambda d, t=t: obj(
                W, B, [a + (d if k == t else 0.0) for k, a in enumerate(xs)], h0))
            self.assertTrue(close(input_grads[t], fd),
                            "input grad t=%d: %r vs %r" % (t, input_grads[t], fd))
        self.assertTrue(close(seq.linear.grad[0],
                              central(lambda d: obj([W[0] + d, W[1]], B, xs, h0))))
        self.assertTrue(close(seq.linear.grad[1],
                              central(lambda d: obj([W[0], W[1] + d], B, xs, h0))))
        self.assertTrue(close(seq.linear.grad_bias,
                              central(lambda d: obj(W, B + d, xs, h0))))
        self.assertTrue(close(grad_initial,
                              central(lambda d: obj(W, B, xs, h0 + d))))

    def test_stream_gradients_no_truncation(self):
        self._check_stream_gradients(XS, None, False, 0.0, 0.0)
        self._check_stream_gradients(XS, None, False, H0, GH)

    def test_stream_gradients_truncated(self):
        self._check_stream_gradients(XS, 1, False, 0.0, 0.0)
        self._check_stream_gradients(XS, 1, True, H0, GH)
        self._check_stream_gradients(XS, 2, False, H0, GH)
        self._check_stream_gradients(XS, 2, True, H0, GH)
        self._check_stream_gradients(XS, 3, True, 0.0, GH)
        self._check_stream_gradients(XS, 100, False, H0, GH)

    def test_stream_plain_backward_matches_batch(self):
        seq, _, _ = self._run_stream(XS, 2, True, H0)
        stream_inputs = seq.backward(GO)
        ref = fresh_sequence(W, B)
        ref.forward(ROWS, 2, True, H0)
        self.assertTrue(allclose(stream_inputs, ref.backward(GO), atol=1e-15))
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad, atol=1e-15))

    def test_empty_session(self):
        seq = fresh_sequence(W, B)
        seq.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        self.assertEqual(seq.finish_stream(), [])
        self.assertTrue(close(seq.hidden, H0, atol=1e-15))
        input_grads, grad_initial = seq.backward_with_initial_hidden([], GH)
        self.assertEqual(input_grads, [])
        # No steps: the terminal hidden state is the initial hidden state.
        self.assertTrue(close(grad_initial, GH, atol=1e-15))
        self.assertEqual(seq.linear.grad, [0.0, 0.0])
        self.assertEqual(seq.backward([]), [])

    def test_single_step_session(self):
        self._check_stream_gradients(XS[:1], None, False, H0, GH)
        self._check_stream_gradients(XS[:1], 1, True, H0, GH)

    def test_carried_boundary_uses_detached_copy(self):
        # Perturbing the upstream gradient of the last step must move steps
        # within its segment only, never across the boundary.
        seq, _, _ = self._run_stream(XS[:4], 2, True, H0)
        base = seq.backward(GO[:4])
        shifted, _, _ = self._run_stream(XS[:4], 2, True, H0)
        moved = shifted.backward([GO[0], GO[1], GO[2], GO[3] + 0.7])
        self.assertTrue(close(moved[0], base[0], atol=1e-15))
        self.assertTrue(close(moved[1], base[1], atol=1e-15))
        self.assertFalse(close(moved[2], base[2], atol=1e-9))
        self.assertFalse(close(moved[3], base[3], atol=1e-9))

    def test_session_uses_forward_time_weights_after_update(self):
        seq, _, returned = self._run_stream(XS, None, False, H0)
        input_grads, _ = seq.backward_with_initial_hidden(GO, GH)
        seq.linear.apply_gradients(0.1)
        # The recorded session keeps its forward-time weights.
        again, _ = seq.backward_with_initial_hidden(GO, GH)
        self.assertTrue(allclose(again, input_grads, atol=1e-15))
        # New sessions follow the updated parameters.
        seq.start_stream(initial_hidden=H0)
        moved_outs = [seq.step([x]) for x in XS]
        seq.finish_stream()
        self.assertTrue(allclose(
            moved_outs,
            sequence_outputs(seq.linear.weight, seq.linear.bias, XS, H0, None, False)))
        self.assertFalse(allclose(moved_outs, returned, atol=1e-9))

    def test_repeated_backward_accumulates(self):
        seq, _, _ = self._run_stream(XS, 2, True, H0)
        seq.backward(GO)
        seq.backward(GO)
        ref = fresh_sequence(W, B)
        ref.forward(ROWS, 2, True, H0)
        ref.backward([2 * g for g in GO])
        self.assertTrue(allclose(seq.linear.grad, ref.linear.grad))
        self.assertTrue(close(seq.linear.grad_bias, ref.linear.grad_bias))

    def test_finish_without_start_raises_and_preserves_state(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        snapshot = (seq.hidden, list(seq.outputs), list(seq.linear.last),
                    list(seq.linear.grad), seq.linear.grad_bias)
        with self.assertRaises(RuntimeError):
            seq.finish_stream()
        self.assertEqual(
            (seq.hidden, list(seq.outputs), list(seq.linear.last),
             list(seq.linear.grad), seq.linear.grad_bias), snapshot)
        # The recorded cache is still fully usable.
        self.assertEqual(len(seq.backward(GO)), len(ROWS))

    def test_double_start_and_double_finish_raise(self):
        seq = fresh_sequence(W, B)
        seq.start_stream(initial_hidden=H0)
        seq.step([XS[0]])
        with self.assertRaises(RuntimeError):
            seq.start_stream()
        # The failed re-start must not clear the in-progress session.
        self.assertEqual(len(seq.outputs), 1)
        self.assertTrue(close(seq.hidden,
                              math.tanh(W[0] * XS[0] + W[1] * H0 + B), atol=1e-15))
        seq.step([XS[1]])
        self.assertEqual(len(seq.finish_stream()), 2)
        with self.assertRaises(RuntimeError):
            seq.finish_stream()
        # The committed cache survives the rejected second finish.
        self.assertEqual(len(seq.backward(GO[:2])), 2)

    def test_backward_during_open_session_raises(self):
        seq = fresh_sequence(W, B)
        seq.start_stream()
        seq.step([XS[0]])
        with self.assertRaises(RuntimeError):
            seq.backward([0.0])
        with self.assertRaises(RuntimeError):
            seq.backward_with_initial_hidden([0.0], GH)
        # The session is still open and finishes normally.
        self.assertEqual(len(seq.finish_stream()), 1)
        self.assertEqual(len(seq.backward([0.0])), 1)

    def test_invalid_start_arguments_raise_and_change_nothing(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        snapshot = (seq.hidden, list(seq.outputs), list(seq.linear.last),
                    list(seq.linear.grad), seq.linear.grad_bias)
        for kwargs in ({"truncate": 0}, {"truncate": -1}, {"truncate": 1.5},
                       {"truncate": True}, {"carry_hidden": 1},
                       {"carry_hidden": None}, {"initial_hidden": True},
                       {"initial_hidden": "0.0"}):
            with self.assertRaises(ValueError):
                seq.start_stream(**kwargs)
            self.assertEqual(
                (seq.hidden, list(seq.outputs), list(seq.linear.last),
                 list(seq.linear.grad), seq.linear.grad_bias), snapshot)
        # No session was opened by the rejected calls; the cache still works.
        self.assertEqual(len(seq.backward(GO)), len(ROWS))
        with self.assertRaises(RuntimeError):
            seq.finish_stream()

    def test_invalid_row_mid_session_changes_nothing_and_session_continues(self):
        seq = fresh_sequence(W, B)
        seq.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        seq.step([XS[0]])
        hidden_before = seq.hidden
        last_before = list(seq.linear.last)
        for bad_row in ([], [1.0, 2.0], ["x"], [True], 7, "ab"):
            with self.assertRaises(ValueError):
                seq.step(bad_row)
        self.assertEqual(seq.hidden, hidden_before)
        self.assertEqual(seq.outputs, [hidden_before])
        self.assertEqual(seq.linear.last, last_before)
        # The session is still alive and completes the same trajectory.
        for x in XS[1:]:
            seq.step([x])
        returned = seq.finish_stream()
        self.assertTrue(allclose(
            returned, sequence_outputs(W, B, XS, H0, 2, True), atol=1e-15))

    def test_step_without_session_keeps_immediate_behavior(self):
        seq = fresh_sequence(W, B)
        seq.forward(ROWS)
        seq.step([0.2])
        with self.assertRaises(RuntimeError):
            seq.backward(GO)
        self.assertTrue(close(seq.outputs[-1],
                              math.tanh(W[0] * 0.2 + W[1] * sequence_outputs(
                                  W, B, XS, 0.0, None, False)[-1] + B), atol=1e-15))

    def test_reset_abandons_open_session(self):
        seq = fresh_sequence(W, B)
        seq.start_stream(initial_hidden=H0)
        seq.step([XS[0]])
        seq.reset()
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])
        with self.assertRaises(RuntimeError):
            seq.finish_stream()
        with self.assertRaises(RuntimeError):
            seq.backward(GO)


class TanhSequenceConstructionTest(unittest.TestCase):
    def test_requires_linear_with_at_least_two_weights(self):
        with self.assertRaises(ValueError):
            TanhSequence(Linear([0.5]))
        with self.assertRaises(ValueError):
            TanhSequence("not a linear")
        with self.assertRaises(ValueError):
            TanhSequence(object())

    def test_accepts_two_or_more_weights(self):
        self.assertEqual(TanhSequence(Linear([0.5, 0.1])).d, 1)
        self.assertEqual(TanhSequence(Linear([0.5, 0.1, 0.2])).d, 2)
        self.assertEqual(TanhSequence(Linear([0.5, 0.1, 0.2, -0.4, 0.7])).d, 4)

    def test_rejected_construction_leaves_linear_untouched(self):
        lin = Linear([0.5], 0.3)
        lin.forward([1.0])
        lin.backward(0.4)
        weight, grads, grad_bias, last = (
            list(lin.weight), list(lin.grad), lin.grad_bias, list(lin.last))
        with self.assertRaises(ValueError):
            TanhSequence(lin)
        self.assertEqual(lin.weight, weight)
        self.assertEqual(lin.grad, grads)
        self.assertEqual(lin.grad_bias, grad_bias)
        self.assertEqual(lin.last, last)


# Multi-feature sample: two input features plus the recurrent weight.
WM = [0.4, -0.3, 0.2]
BM = 0.15
ROWSM = [[0.8, -0.2], [-0.5, 1.1], [1.2, 0.4], [-0.7, -0.9], [0.3, 0.6]]
GOM = [0.3, -0.6, 0.9, -0.2, 0.5]


def sequence_outputs_rows(weight, bias, rows, h0, truncate, carry, frozen=None):
    """tanh(sum_k w_k*x_k + w_rec*h_prev + b) over multi-feature rows."""
    d = len(weight) - 1
    outs = []
    hidden = h0
    for i, row in enumerate(rows):
        if truncate is not None and i % truncate == 0:
            if i == 0:
                hidden = h0
            elif frozen is not None:
                hidden = frozen[i]
            elif carry:
                hidden = outs[-1]
            else:
                hidden = 0.0
        pre = sum(weight[k] * row[k] for k in range(d)) + weight[d] * hidden + bias
        hidden = math.tanh(pre)
        outs.append(hidden)
    return outs


def frozen_boundaries_rows(rows, truncate, carry, weight, bias, h0):
    if truncate is None:
        return {}
    base = sequence_outputs_rows(weight, bias, rows, h0, truncate, carry)
    frozen = {}
    for i in range(1, len(rows)):
        if i % truncate == 0:
            frozen[i] = base[i - 1] if carry else 0.0
    return frozen


def objective_rows(weight, bias, rows, h0, grad_outputs, grad_hidden,
                   truncate, carry, frozen):
    outs = sequence_outputs_rows(weight, bias, rows, h0, truncate, carry, frozen)
    if not outs:
        return grad_hidden * h0
    return sum(g * o for g, o in zip(grad_outputs, outs)) + grad_hidden * outs[-1]


class MultiFeatureSequenceTest(unittest.TestCase):
    D = len(WM) - 1

    def fresh(self):
        return TanhSequence(Linear(list(WM), BM))

    def test_forward_matches_reference_formula(self):
        for truncate, carry, h0 in ((None, False, 0.0), (None, False, H0),
                                    (1, False, 0.0), (1, True, H0),
                                    (2, False, 0.0), (2, True, H0),
                                    (3, True, 0.0), (100, False, H0)):
            seq = self.fresh()
            outs = seq.forward(ROWSM, truncate, carry, h0)
            expected = sequence_outputs_rows(WM, BM, ROWSM, h0, truncate, carry)
            self.assertTrue(allclose(outs, expected, atol=1e-12),
                            "truncate=%r carry=%r h0=%r" % (truncate, carry, h0))
            self.assertTrue(close(seq.hidden, expected[-1], atol=1e-12))
            self.assertEqual(seq.outputs, outs)

    def test_empty_forward_and_gradients(self):
        seq = self.fresh()
        self.assertEqual(seq.forward([], truncate=2, carry_hidden=True,
                                     initial_hidden=H0), [])
        self.assertTrue(close(seq.hidden, H0, atol=1e-15))
        input_grads, grad_initial = seq.backward_with_initial_hidden([], GH)
        self.assertEqual(input_grads, [])
        self.assertTrue(close(grad_initial, GH, atol=1e-15))
        self.assertEqual(seq.linear.grad, [0.0] * len(WM))
        self.assertEqual(seq.backward([]), [])

    def _check_gradients(self, rows, truncate, carry, h0, grad_hidden):
        go = GOM[:len(rows)]
        frozen = frozen_boundaries_rows(rows, truncate, carry, WM, BM, h0)
        seq = self.fresh()
        seq.forward(rows, truncate, carry, h0)
        input_grads, grad_initial = seq.backward_with_initial_hidden(go, grad_hidden)

        def obj(weight, bias, inputs, initial):
            return objective_rows(weight, bias, inputs, initial, go, grad_hidden,
                                  truncate, carry, frozen)

        # Input gradients mirror the input shape: one row of d gradients per
        # step, each matching the finite difference of that feature.
        self.assertEqual(len(input_grads), len(rows))
        for t in range(len(rows)):
            self.assertEqual(len(input_grads[t]), self.D)
            for k in range(self.D):
                fd = central(lambda dd, t=t, k=k: obj(
                    WM, BM,
                    [[v + (dd if j == k else 0.0) for j, v in enumerate(row)]
                     if i == t else list(row) for i, row in enumerate(rows)],
                    h0))
                self.assertTrue(close(input_grads[t][k], fd),
                                "input grad t=%d k=%d: %r vs %r"
                                % (t, k, input_grads[t][k], fd))
        # Every weight (input coefficients and the recurrent coefficient) and
        # the bias accumulate on the same Linear.
        for k in range(len(WM)):
            fd = central(lambda dd, k=k: obj(
                [w + (dd if j == k else 0.0) for j, w in enumerate(WM)],
                BM, rows, h0))
            self.assertTrue(close(seq.linear.grad[k], fd),
                            "weight grad %d: %r vs %r" % (k, seq.linear.grad[k], fd))
        self.assertTrue(close(seq.linear.grad_bias,
                              central(lambda dd: obj(WM, BM + dd, rows, h0))))
        self.assertTrue(close(grad_initial,
                              central(lambda dd: obj(WM, BM, rows, h0 + dd))))
        return seq, input_grads

    def test_backward_matches_finite_differences(self):
        self._check_gradients(ROWSM, None, False, 0.0, 0.0)
        self._check_gradients(ROWSM, None, False, H0, GH)
        self._check_gradients(ROWSM, 1, False, 0.0, GH)
        self._check_gradients(ROWSM, 1, True, H0, GH)
        self._check_gradients(ROWSM, 2, False, H0, 0.0)
        self._check_gradients(ROWSM, 2, True, H0, GH)
        self._check_gradients(ROWSM, 3, True, 0.0, GH)
        self._check_gradients(ROWSM, 100, False, H0, GH)
        self._check_gradients(ROWSM[:1], None, False, H0, GH)

    def test_plain_backward_returns_nested_rows(self):
        seq = self.fresh()
        seq.forward(ROWSM, truncate=2, carry_hidden=True, initial_hidden=H0)
        input_grads = seq.backward(GOM)
        self.assertEqual(len(input_grads), len(ROWSM))
        for row_grads in input_grads:
            self.assertIsInstance(row_grads, list)
            self.assertEqual(len(row_grads), self.D)
        # backward() with a zero terminal seed matches the seeded entry.
        twin = self.fresh()
        twin.forward(ROWSM, truncate=2, carry_hidden=True, initial_hidden=H0)
        seeded, _ = twin.backward_with_initial_hidden(GOM, 0.0)
        for a, b in zip(input_grads, seeded):
            self.assertTrue(allclose(a, b, atol=1e-15))

    def test_step_and_stream_match_batch(self):
        # A plain stepwise walk never truncates; it matches the untruncated
        # batch traversal from the same initial hidden state.
        untruncated = self.fresh()
        untruncated_outs = untruncated.forward(ROWSM, None, False, H0)
        stepped = self.fresh()
        stepped.forward([], initial_hidden=H0)
        step_outs = [stepped.step(list(row)) for row in ROWSM]
        self.assertTrue(allclose(step_outs, untruncated_outs, atol=1e-15))
        self.assertTrue(close(stepped.hidden, untruncated.hidden, atol=1e-15))

        for truncate, carry in ((None, False), (1, True), (2, True), (3, False)):
            batched = self.fresh()
            batch_outs = batched.forward(ROWSM, truncate, carry, H0)

            streamed = self.fresh()
            streamed.start_stream(initial_hidden=H0, truncate=truncate,
                                  carry_hidden=carry)
            stream_outs = [streamed.step(list(row)) for row in ROWSM]
            returned = streamed.finish_stream()
            self.assertTrue(allclose(stream_outs, batch_outs, atol=1e-15))
            self.assertTrue(allclose(returned, batch_outs, atol=1e-15))

            # The finished session back-propagates exactly like the batch pass.
            batch_inputs, batch_initial = batched.backward_with_initial_hidden(GOM, GH)
            stream_inputs, stream_initial = streamed.backward_with_initial_hidden(GOM, GH)
            for a, b in zip(stream_inputs, batch_inputs):
                self.assertTrue(allclose(a, b, atol=1e-15))
            self.assertTrue(close(stream_initial, batch_initial, atol=1e-15))
            self.assertTrue(allclose(streamed.linear.grad, batched.linear.grad,
                                     atol=1e-15))
            self.assertTrue(close(streamed.linear.grad_bias,
                                  batched.linear.grad_bias, atol=1e-15))

    def test_truncated_stream_severs_gradient_at_boundaries(self):
        go = GOM[:4]
        base = self.fresh()
        base.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        for row in ROWSM[:4]:
            base.step(list(row))
        base.finish_stream()
        base_grads = base.backward(go)

        shifted = self.fresh()
        shifted.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        for row in ROWSM[:4]:
            shifted.step(list(row))
        shifted.finish_stream()
        moved = shifted.backward([go[0], go[1], go[2], go[3] + 0.7])
        self.assertTrue(allclose(moved[0], base_grads[0], atol=1e-15))
        self.assertTrue(allclose(moved[1], base_grads[1], atol=1e-15))
        self.assertFalse(allclose(moved[2], base_grads[2], atol=1e-9))
        self.assertFalse(allclose(moved[3], base_grads[3], atol=1e-9))

    def test_invalid_rows_raise_and_roll_back_all_state(self):
        seq = self.fresh()
        seq.forward(ROWSM, truncate=2)
        seq.backward(GOM)
        snapshot = (seq.hidden, list(seq.outputs), list(seq.linear.last),
                    list(seq.linear.grad), seq.linear.grad_bias)
        bad_row_sets = (
            [[1.0]],                          # too narrow
            [[1.0, 2.0, 3.0]],                # too wide
            [[1.0, 2.0], [1.0]],              # bad width mid-traversal
            [[1.0, "x"]],                     # non-numeric element
            [[True, 1.0]],                    # bool element
            [[1.0, 2.0], None],               # non-sequence row
            [[1.0, 2.0], 7],                  # scalar row
            ["ab"],                           # text row
            [b"ab"],                          # bytes row
        )
        for bad_rows in bad_row_sets:
            with self.assertRaises(ValueError):
                seq.forward(bad_rows, truncate=2)
            self.assertEqual(seq.hidden, snapshot[0])
            self.assertEqual(seq.outputs, snapshot[1])
            self.assertEqual(seq.linear.last, snapshot[2])
            self.assertEqual(seq.linear.grad, snapshot[3])
            self.assertEqual(seq.linear.grad_bias, snapshot[4])
        # The pre-error cache is still fully back-propagatable.
        self.assertEqual(len(seq.backward(GOM)), len(ROWSM))

    def test_invalid_step_row_changes_nothing(self):
        seq = self.fresh()
        seq.forward(ROWSM[:2])
        hidden_before, outputs_before = seq.hidden, list(seq.outputs)
        last_before = list(seq.linear.last)
        for bad_row in ([1.0], [1.0, 2.0, 3.0], [1.0, "x"], [True, 0.5],
                        "ab", b"ab", 7, None):
            with self.assertRaises(ValueError):
                seq.step(bad_row)
        self.assertEqual(seq.hidden, hidden_before)
        self.assertEqual(seq.outputs, outputs_before)
        self.assertEqual(seq.linear.last, last_before)
        # The batch cache survived every rejected step.
        self.assertEqual(len(seq.backward(GOM[:2])), 2)

    def test_invalid_row_mid_session_changes_nothing_and_session_continues(self):
        seq = self.fresh()
        seq.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        seq.step(list(ROWSM[0]))
        hidden_before = seq.hidden
        last_before = list(seq.linear.last)
        for bad_row in ([1.0], [1.0, 2.0, 3.0], ["x", 1.0], [True, 1.0],
                        7, "ab"):
            with self.assertRaises(ValueError):
                seq.step(bad_row)
        self.assertEqual(seq.hidden, hidden_before)
        self.assertEqual(seq.outputs, [hidden_before])
        self.assertEqual(seq.linear.last, last_before)
        for row in ROWSM[1:]:
            seq.step(list(row))
        returned = seq.finish_stream()
        self.assertTrue(allclose(
            returned, sequence_outputs_rows(WM, BM, ROWSM, H0, 2, True),
            atol=1e-15))

    def test_scalar_sequence_behavior_is_unchanged(self):
        # d == 1 keeps the flat scalar gradient list, not nested rows.
        seq = fresh_sequence(W, B)
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        input_grads = seq.backward(GO)
        self.assertEqual(len(input_grads), len(ROWS))
        for grad in input_grads:
            self.assertIsInstance(grad, float)


if __name__ == "__main__":
    unittest.main()
