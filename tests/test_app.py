"""Regression suite for the Linear / TanhSequence CPU sequence core.

Standard library only (unittest + math). Numerical gradients are checked
with central finite differences against explicit absolute/relative
tolerances; trajectories are checked against an independent in-test
reference implementation of the documented formulas.
"""

import math
import unittest

from app import Linear, TanhSequence

# Tolerances for exact-formula comparisons (same operations, reordering only).
EXACT_ABS = 1e-12
EXACT_REL = 1e-12
# Tolerances for central finite-difference comparisons.
FD_ABS = 1e-6
FD_REL = 1e-6
FD_H = 1e-6


def central_diff(f, x, h=FD_H):
    """Central finite difference of scalar f at x."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def reference_outputs(weight, bias, rows, truncate, carry_hidden, initial_hidden):
    """Independent reimplementation of the documented TanhSequence.forward
    traversal: tanh(w0 * x + w1 * h_prev + b), with segment starts at
    i % truncate == 0 resetting the hidden state to zero, or to a numeric
    copy of the previous segment's last hidden value when carrying."""
    hidden = initial_hidden
    outputs = []
    for i, x in enumerate(rows):
        if truncate is not None and i % truncate == 0:
            if i > 0:
                hidden = outputs[-1] if carry_hidden else 0.0
            else:
                hidden = initial_hidden
        hidden = math.tanh(weight[0] * x + weight[1] * hidden + bias)
        outputs.append(hidden)
    return outputs


def sequence_loss(weight, bias, rows, truncate, carry_hidden, initial_hidden,
                  coeffs, grad_hidden):
    """Build the scalar loss L = sum(coeffs[t] * out[t]) + grad_hidden *
    final_hidden whose analytic gradient is exactly truncated BPTT.

    Carried hidden values are detached constants (numeric copies of the
    unperturbed trajectory's segment-end values), mirroring the documented
    segment-boundary semantics: they enter the local derivatives of the
    segment's first step but let no gradient cross back. The public
    forward's outputs are cross-checked against reference_outputs
    separately; this oracle is what backward() is finite-differenced
    against."""
    base = reference_outputs(weight, bias, rows, truncate, carry_hidden,
                             initial_hidden)

    def evaluate(w, b, xs, ih):
        hidden = ih
        outs = []
        for i, x in enumerate(xs):
            if truncate is not None and i % truncate == 0:
                if i > 0:
                    hidden = base[i - 1] if carry_hidden else 0.0
                else:
                    hidden = ih
            hidden = math.tanh(w[0] * x + w[1] * hidden + b)
            outs.append(hidden)
        total = sum(c * o for c, o in zip(coeffs, outs))
        # The terminal hidden state after an empty traversal is initial_hidden.
        total += grad_hidden * (outs[-1] if outs else ih)
        return total

    return evaluate


class CloseMixin:
    def assertClose(self, actual, expected, abs_tol=EXACT_ABS, rel_tol=EXACT_REL):
        if abs(actual - expected) > max(abs_tol, rel_tol * abs(expected)):
            raise self.failureException(
                "%r != %r within abs_tol=%r rel_tol=%r"
                % (actual, expected, abs_tol, rel_tol))

    def assertListClose(self, actual, expected, abs_tol=EXACT_ABS, rel_tol=EXACT_REL):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertClose(a, e, abs_tol, rel_tol)


class SmokeTest(unittest.TestCase):
    def test_import(self):
        import app
        self.assertTrue(app)


# ---------------------------------------------------------------------------
# Linear: forward
# ---------------------------------------------------------------------------

class LinearForwardTest(CloseMixin, unittest.TestCase):
    def test_forward_matches_public_formula(self):
        lin = Linear([0.5, -0.25, 2.0], bias=0.1)
        result = lin.forward([2.0, 4.0, -1.0])
        self.assertClose(result, 0.5 * 2.0 - 0.25 * 4.0 + 2.0 * -1.0 + 0.1)

    def test_forward_accepts_ints_and_tuples(self):
        lin = Linear((1, 2), bias=1)
        self.assertClose(lin.forward((3, 4)), 1 * 3 + 2 * 4 + 1)

    def test_forward_default_bias_is_zero(self):
        lin = Linear([2.0])
        self.assertClose(lin.bias, 0.0)
        self.assertClose(lin.forward([1.5]), 3.0)

    def test_forward_does_not_mutate_parameters(self):
        lin = Linear([0.3, -0.7], bias=0.2)
        lin.forward([1.0, 2.0])
        self.assertListClose(lin.weight, [0.3, -0.7])
        self.assertClose(lin.bias, 0.2)


# ---------------------------------------------------------------------------
# Linear: backward, accumulation, zero_grad, apply_gradients
# ---------------------------------------------------------------------------

class LinearBackwardTest(CloseMixin, unittest.TestCase):
    def setUp(self):
        self.weight = [0.4, -0.6, 0.9]
        self.bias = 0.15
        self.x = [1.2, -0.8, 0.5]
        self.coeff = 0.7  # upstream gradient: loss = coeff * forward(x)

    def make(self):
        return Linear(list(self.weight), self.bias)

    def loss(self, weight, bias, x):
        return self.coeff * self.make_with(weight, bias).forward(x)

    def make_with(self, weight, bias):
        lin = Linear(list(weight), bias)
        return lin

    def test_input_gradient_matches_finite_differences(self):
        lin = self.make()
        lin.forward(list(self.x))
        input_grads = lin.backward(self.coeff)
        for i in range(len(self.x)):
            def f(xi, i=i):
                x = list(self.x)
                x[i] = xi
                return self.coeff * self.make().forward(x)
            self.assertClose(input_grads[i], central_diff(f, self.x[i]),
                             FD_ABS, FD_REL)

    def test_weight_and_bias_gradients_match_finite_differences(self):
        lin = self.make()
        lin.forward(list(self.x))
        lin.backward(self.coeff)
        for i in range(len(self.weight)):
            def f(wi, i=i):
                w = list(self.weight)
                w[i] = wi
                return self.coeff * self.make_with(w, self.bias).forward(self.x)
            self.assertClose(lin.grad[i], central_diff(f, self.weight[i]),
                             FD_ABS, FD_REL)

        def fb(b):
            return self.coeff * self.make_with(self.weight, b).forward(self.x)
        self.assertClose(lin.grad_bias, central_diff(fb, self.bias),
                         FD_ABS, FD_REL)

    def test_repeated_backward_accumulates_instead_of_overwriting(self):
        lin = self.make()
        lin.forward(list(self.x))
        lin.backward(0.5)
        lin.backward(1.5)
        # grad = (0.5 + 1.5) * x, grad_bias = 0.5 + 1.5
        self.assertListClose(lin.grad, [2.0 * v for v in self.x])
        self.assertClose(lin.grad_bias, 2.0)

    def test_backward_uses_weights_from_forward_time(self):
        lin = self.make()
        lin.forward(list(self.x))
        expected = [self.coeff * w for w in self.weight]
        # Change the parameters between forward and backward.
        lin.weight = [9.0, 9.0, 9.0]
        input_grads = lin.backward(self.coeff)
        self.assertListClose(input_grads, expected)

    def test_zero_grad_clears_only_gradients(self):
        lin = self.make()
        lin.forward(list(self.x))
        lin.backward(self.coeff)
        lin.zero_grad()
        self.assertListClose(lin.grad, [0.0] * len(self.weight))
        self.assertClose(lin.grad_bias, 0.0)
        # Parameters and the cached forward record are untouched.
        self.assertListClose(lin.weight, self.weight)
        self.assertClose(lin.bias, self.bias)
        input_grads = lin.backward(self.coeff)
        self.assertListClose(input_grads, [self.coeff * w for w in self.weight])

    def test_apply_gradients_subtracts_learning_rate_times_gradient(self):
        lin = self.make()
        lin.forward(list(self.x))
        lin.backward(self.coeff)
        grads = list(lin.grad)
        grad_bias = lin.grad_bias
        lr = 0.25
        lin.apply_gradients(lr)
        expected_w = [w - lr * g for w, g in zip(self.weight, grads)]
        expected_b = self.bias - lr * grad_bias
        self.assertListClose(lin.weight, expected_w)
        self.assertClose(lin.bias, expected_b)
        # The updated parameters drive the next forward pass.
        out = lin.forward(list(self.x))
        self.assertClose(out, sum(w * a for w, a in zip(expected_w, self.x))
                         + expected_b)

    def test_apply_gradients_result_matches_finite_difference_descent(self):
        # One descent step on L = 0.5 * forward(x)^2 must reduce the loss,
        # and the updated parameters must satisfy the public update formula.
        lin = self.make()
        target = 0.0
        out = lin.forward(list(self.x))
        lin.backward(out - target)  # dL/dout for L = 0.5 * (out - target)^2
        grads = list(lin.grad)
        grad_bias = lin.grad_bias
        lr = 0.1
        before = 0.5 * (out - target) ** 2
        lin.apply_gradients(lr)
        self.assertListClose(lin.weight,
                             [w - lr * g for w, g in zip(self.weight, grads)])
        self.assertClose(lin.bias, self.bias - lr * grad_bias)
        after = 0.5 * (lin.forward(list(self.x)) - target) ** 2
        self.assertLess(after, before)


# ---------------------------------------------------------------------------
# Linear: error contract
# ---------------------------------------------------------------------------

class LinearErrorTest(CloseMixin, unittest.TestCase):
    def test_constructor_rejects_non_numeric_and_bool(self):
        for bad_weight in ("abc", [1.0, "x"], [1.0, True], [1.0, None]):
            with self.assertRaises(ValueError, msg=repr(bad_weight)):
                Linear(bad_weight)
        for bad_bias in ("x", True, None, [0.1]):
            with self.assertRaises(ValueError, msg=repr(bad_bias)):
                Linear([1.0], bias=bad_bias)

    def test_forward_rejects_bad_input_and_keeps_previous_cache(self):
        lin = Linear([0.5, -0.5], bias=0.1)
        lin.forward([2.0, 3.0])
        for bad_x in ([1.0], [1.0, 2.0, 3.0], [1.0, "x"], [1.0, True], "ab"):
            with self.assertRaises(ValueError, msg=repr(bad_x)):
                lin.forward(bad_x)
        # The failed calls left the earlier successful forward cached.
        input_grads = lin.backward(1.0)
        self.assertListClose(input_grads, [0.5, -0.5])
        self.assertListClose(lin.grad, [2.0, 3.0])
        self.assertClose(lin.grad_bias, 1.0)

    def test_backward_without_forward_raises_runtime_error(self):
        lin = Linear([1.0, 2.0])
        with self.assertRaises(RuntimeError):
            lin.backward(1.0)

    def test_backward_rejects_non_numeric_and_bool_grad(self):
        lin = Linear([1.0, 2.0])
        lin.forward([1.0, 1.0])
        for bad_grad in ("x", True, None, [1.0]):
            with self.assertRaises(ValueError, msg=repr(bad_grad)):
                lin.backward(bad_grad)
        # Failed calls accumulated nothing.
        self.assertListClose(lin.grad, [0.0, 0.0])
        self.assertClose(lin.grad_bias, 0.0)

    def test_apply_gradients_rejects_non_numeric_learning_rate(self):
        lin = Linear([1.0, 2.0], bias=0.5)
        lin.forward([1.0, 1.0])
        lin.backward(1.0)
        for bad_lr in ("x", True, None):
            with self.assertRaises(ValueError, msg=repr(bad_lr)):
                lin.apply_gradients(bad_lr)
        self.assertListClose(lin.weight, [1.0, 2.0])
        self.assertClose(lin.bias, 0.5)


# ---------------------------------------------------------------------------
# TanhSequence: trajectory (step vs batch forward), empty and single-step
# ---------------------------------------------------------------------------

class SequenceTrajectoryTest(CloseMixin, unittest.TestCase):
    WEIGHT = [0.4, 0.2]
    BIAS = 0.05
    ROWS = [0.5, -0.3, 1.0, 0.7]

    def make_seq(self):
        return TanhSequence(Linear(list(self.WEIGHT), self.BIAS))

    def test_step_and_batch_forward_produce_same_trajectory(self):
        step_seq = self.make_seq()
        stepped = [step_seq.step([x]) for x in self.ROWS]
        batch_seq = self.make_seq()
        batched = batch_seq.forward([[x] for x in self.ROWS])
        self.assertListClose(stepped, batched)
        self.assertListClose(step_seq.outputs, batched)
        self.assertClose(step_seq.hidden, batch_seq.hidden)

    def test_step_matches_public_formula_and_updates_state(self):
        seq = self.make_seq()
        expected_hidden = 0.0
        for x in self.ROWS:
            expected_hidden = math.tanh(
                self.WEIGHT[0] * x + self.WEIGHT[1] * expected_hidden + self.BIAS)
            returned = seq.step([x])
            self.assertClose(returned, expected_hidden)
            self.assertClose(seq.hidden, expected_hidden)
            self.assertClose(seq.outputs[-1], expected_hidden)

    def test_step_continues_trajectory_after_batch_forward(self):
        seq = self.make_seq()
        seq.forward([[self.ROWS[0]], [self.ROWS[1]]])
        hidden = seq.hidden
        out = seq.step([self.ROWS[2]])
        self.assertClose(out, math.tanh(
            self.WEIGHT[0] * self.ROWS[2] + self.WEIGHT[1] * hidden + self.BIAS))
        self.assertEqual(len(seq.outputs), 3)

    def test_empty_forward_commits_initial_hidden_and_empty_outputs(self):
        seq = self.make_seq()
        returned = seq.forward([])
        self.assertEqual(returned, [])
        self.assertEqual(seq.outputs, [])
        self.assertClose(seq.hidden, 0.0)
        seq2 = self.make_seq()
        seq2.forward([], initial_hidden=0.4)
        self.assertClose(seq2.hidden, 0.4)
        self.assertEqual(seq2.outputs, [])

    def test_single_step_forward(self):
        seq = self.make_seq()
        out = seq.forward([[0.9]])
        self.assertEqual(len(out), 1)
        self.assertClose(out[0], math.tanh(self.WEIGHT[0] * 0.9 + self.BIAS))
        self.assertClose(seq.hidden, out[0])


# ---------------------------------------------------------------------------
# TanhSequence: forward traversal under every truncation/carry combination
# ---------------------------------------------------------------------------

class SequenceForwardTruncationTest(CloseMixin, unittest.TestCase):
    WEIGHT = [0.45, -0.35]
    BIAS = 0.08
    ROWS = [0.3, -0.7, 1.1, 0.2, -0.5]

    def run_case(self, rows, truncate, carry_hidden, initial_hidden=0.0):
        seq = TanhSequence(Linear(list(self.WEIGHT), self.BIAS))
        out = seq.forward([[x] for x in rows], truncate=truncate,
                          carry_hidden=carry_hidden,
                          initial_hidden=initial_hidden)
        expected = reference_outputs(self.WEIGHT, self.BIAS, rows, truncate,
                                     carry_hidden, initial_hidden)
        self.assertListClose(out, expected)
        self.assertListClose(seq.outputs, expected)
        if expected:
            self.assertClose(seq.hidden, expected[-1])
        else:
            self.assertClose(seq.hidden, initial_hidden)

    def test_no_truncation(self):
        self.run_case(self.ROWS, None, False)
        self.run_case(self.ROWS, None, False, initial_hidden=0.3)

    def test_truncate_one_resets_every_step(self):
        self.run_case(self.ROWS, 1, False)
        # Without carry every step starts from zero hidden state.
        seq = TanhSequence(Linear(list(self.WEIGHT), self.BIAS))
        out = seq.forward([[x] for x in self.ROWS], truncate=1)
        for x, o in zip(self.ROWS, out):
            self.assertClose(o, math.tanh(self.WEIGHT[0] * x + self.BIAS))

    def test_truncate_one_with_carry_equals_no_truncation(self):
        # Carrying across length-1 segments passes the detached previous
        # hidden value, so the trajectory matches the untruncated one.
        self.run_case(self.ROWS, 1, True)
        plain = reference_outputs(self.WEIGHT, self.BIAS, self.ROWS, None, False, 0.0)
        carried = reference_outputs(self.WEIGHT, self.BIAS, self.ROWS, 1, True, 0.0)
        self.assertListClose(carried, plain)

    def test_truncate_in_middle(self):
        self.run_case(self.ROWS, 2, False)
        self.run_case(self.ROWS, 2, True)
        self.run_case(self.ROWS, 3, False)
        self.run_case(self.ROWS, 3, True)

    def test_truncate_longer_than_sequence(self):
        self.run_case(self.ROWS, 10, False)
        self.run_case(self.ROWS, 10, True)
        # Only the i == 0 boundary exists, so this matches no truncation.
        plain = reference_outputs(self.WEIGHT, self.BIAS, self.ROWS, None, False, 0.0)
        long_trunc = reference_outputs(self.WEIGHT, self.BIAS, self.ROWS, 10, True, 0.0)
        self.assertListClose(long_trunc, plain)

    def test_segment_start_hidden_values(self):
        # carry_hidden=True: each new segment starts from a numeric copy of
        # the previous segment's last hidden value; False: from zero.
        rows = self.ROWS[:4]
        truncate = 2
        for carry, start in ((True, None), (False, 0.0)):
            seq = TanhSequence(Linear(list(self.WEIGHT), self.BIAS))
            out = seq.forward([[x] for x in rows], truncate=truncate,
                              carry_hidden=carry)
            boundary_hidden = out[truncate - 1] if carry else 0.0
            self.assertClose(
                out[truncate],
                math.tanh(self.WEIGHT[0] * rows[truncate]
                          + self.WEIGHT[1] * boundary_hidden + self.BIAS))

    def test_forward_returns_current_outputs_list(self):
        seq = TanhSequence(Linear(list(self.WEIGHT), self.BIAS))
        returned = seq.forward([[x] for x in self.ROWS])
        self.assertIs(returned, seq.outputs)


# ---------------------------------------------------------------------------
# TanhSequence: backward verified by central finite differences
# ---------------------------------------------------------------------------

class SequenceBackwardFiniteDifferenceTest(CloseMixin, unittest.TestCase):
    WEIGHT = [0.45, -0.35]
    BIAS = 0.08
    INIT_H = 0.2
    GRAD_HIDDEN = 0.15

    def check_case(self, rows, truncate, carry_hidden):
        coeffs = [0.35, -0.6, 0.9, 0.25, -0.45][:len(rows)]
        init_h = self.INIT_H
        grad_hidden = self.GRAD_HIDDEN

        lin = Linear(list(self.WEIGHT), self.BIAS)
        seq = TanhSequence(lin)
        seq.forward([[x] for x in rows], truncate=truncate,
                    carry_hidden=carry_hidden, initial_hidden=init_h)
        input_grads, grad_init = seq.backward_with_initial_hidden(
            list(coeffs), grad_hidden)
        self.assertEqual(len(input_grads), len(rows))

        evaluate = sequence_loss(self.WEIGHT, self.BIAS, rows, truncate,
                                 carry_hidden, init_h, coeffs, grad_hidden)

        def loss(w=None, b=None, xs=None, ih=None):
            return evaluate(
                w if w is not None else self.WEIGHT,
                b if b is not None else self.BIAS,
                xs if xs is not None else rows,
                ih if ih is not None else init_h)

        # dL/d(input_t)
        for t in range(len(rows)):
            def f(xt, t=t):
                xs = list(rows)
                xs[t] = xt
                return loss(xs=xs)
            self.assertClose(input_grads[t], central_diff(f, rows[t]),
                             FD_ABS, FD_REL)
        # dL/d(weight_i) accumulated on the Linear
        for i in range(2):
            def f(wi, i=i):
                w = list(self.WEIGHT)
                w[i] = wi
                return loss(w=w)
            self.assertClose(lin.grad[i], central_diff(f, self.WEIGHT[i]),
                             FD_ABS, FD_REL)
        # dL/d(bias)
        self.assertClose(lin.grad_bias,
                         central_diff(lambda b: loss(b=b), self.BIAS),
                         FD_ABS, FD_REL)
        # dL/d(initial_hidden)
        self.assertClose(grad_init,
                         central_diff(lambda ih: loss(ih=ih), init_h),
                         FD_ABS, FD_REL)

    def test_no_truncation(self):
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], None, False)

    def test_truncate_one(self):
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 1, False)
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 1, True)

    def test_truncate_in_middle(self):
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 2, False)
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 2, True)
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 3, True)

    def test_truncate_longer_than_sequence(self):
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 10, False)
        self.check_case([0.3, -0.7, 1.1, 0.2, -0.5], 10, True)

    def test_single_step(self):
        self.check_case([0.9], None, False)
        self.check_case([0.9], 1, True)

    def test_empty_sequence_backward(self):
        lin = Linear(list(self.WEIGHT), self.BIAS)
        seq = TanhSequence(lin)
        seq.forward([], initial_hidden=self.INIT_H)
        input_grads, grad_init = seq.backward_with_initial_hidden(
            [], self.GRAD_HIDDEN)
        self.assertEqual(input_grads, [])
        # No steps: the terminal gradient passes through unchanged and no
        # parameter gradient is accumulated.
        self.assertClose(grad_init, self.GRAD_HIDDEN)
        self.assertListClose(lin.grad, [0.0, 0.0])
        self.assertClose(lin.grad_bias, 0.0)

    def test_public_entry_finite_difference_no_truncation(self):
        # End-to-end through the public entry points only: without
        # truncation there is no detached boundary, so the numerical
        # derivative of the public forward loss must match backward().
        rows = [0.3, -0.7, 1.1]
        coeffs = [0.5, -0.4, 0.8]

        def public_loss(xs):
            seq = TanhSequence(Linear(list(self.WEIGHT), self.BIAS))
            outs = seq.forward([[x] for x in xs])
            return sum(c * o for c, o in zip(coeffs, outs))

        lin = Linear(list(self.WEIGHT), self.BIAS)
        seq = TanhSequence(lin)
        seq.forward([[x] for x in rows])
        input_grads = seq.backward(list(coeffs))
        for t in range(len(rows)):
            def f(xt, t=t):
                xs = list(rows)
                xs[t] = xt
                return public_loss(xs)
            self.assertClose(input_grads[t], central_diff(f, rows[t]),
                             FD_ABS, FD_REL)

    def test_plain_backward_matches_zero_terminal_seed(self):
        rows = [0.3, -0.7, 1.1]
        coeffs = [0.5, -0.4, 0.8]
        results = []
        for use_seed in (False, True):
            lin = Linear(list(self.WEIGHT), self.BIAS)
            seq = TanhSequence(lin)
            seq.forward([[x] for x in rows])
            if use_seed:
                in_grads, _ = seq.backward_with_initial_hidden(coeffs, 0.0)
            else:
                in_grads = seq.backward(coeffs)
            results.append((in_grads, list(lin.grad), lin.grad_bias))
        self.assertListClose(results[0][0], results[1][0])
        self.assertListClose(results[0][1], results[1][1])
        self.assertClose(results[0][2], results[1][2])

    def test_backward_returns_input_gradient_list_only(self):
        seq = TanhSequence(Linear(list(self.WEIGHT), self.BIAS))
        seq.forward([[0.2], [0.4]])
        result = seq.backward([1.0, 1.0])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# TanhSequence: truncation boundary gradient semantics
# ---------------------------------------------------------------------------

class SequenceBoundaryGradientTest(CloseMixin, unittest.TestCase):
    WEIGHT = [0.45, -0.35]
    BIAS = 0.08
    ROWS = [0.3, -0.7, 1.1, 0.2]

    def make_seq(self):
        return TanhSequence(Linear(list(self.WEIGHT), self.BIAS))

    def test_no_gradient_crosses_boundary_regardless_of_carry(self):
        for carry in (False, True):
            seq = self.make_seq()
            seq.forward([[x] for x in self.ROWS], truncate=2,
                        carry_hidden=carry)
            # Loss depends only on the second segment's outputs.
            input_grads = seq.backward([0.0, 0.0, 0.9, -0.6])
            self.assertEqual(input_grads[0], 0.0)
            self.assertEqual(input_grads[1], 0.0)
            self.assertNotEqual(input_grads[2], 0.0)
            self.assertNotEqual(input_grads[3], 0.0)

    def test_initial_hidden_gradient_severed_by_boundary(self):
        for carry in (False, True):
            seq = self.make_seq()
            seq.forward([[x] for x in self.ROWS], truncate=2,
                        carry_hidden=carry, initial_hidden=0.3)
            _, grad_init = seq.backward_with_initial_hidden(
                [0.0, 0.0, 0.9, -0.6], 0.0)
            self.assertEqual(grad_init, 0.0)

    def test_segment_start_local_parameter_gradients_are_kept(self):
        # The carried hidden value enters the segment start's local
        # derivative as a constant: grad[1] picks up d_pre * carried_hidden.
        seq = self.make_seq()
        out = seq.forward([[x] for x in self.ROWS], truncate=2,
                          carry_hidden=True)
        seq.backward([0.0, 0.0, 1.0, 0.0])
        d_pre_2 = 1.0 * (1.0 - out[2] * out[2])
        self.assertClose(seq.linear.grad[0], d_pre_2 * self.ROWS[2])
        self.assertClose(seq.linear.grad[1], d_pre_2 * out[1])  # carried copy
        self.assertClose(seq.linear.grad_bias, d_pre_2)

    def test_gradients_within_segment_are_preserved(self):
        # A loss on the last step of a segment still reaches earlier inputs
        # of the same segment.
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS], truncate=3)
        input_grads = seq.backward([0.0, 0.0, 1.0, 0.0])
        self.assertNotEqual(input_grads[0], 0.0)
        self.assertNotEqual(input_grads[1], 0.0)
        self.assertNotEqual(input_grads[2], 0.0)
        self.assertEqual(input_grads[3], 0.0)  # next segment, zero upstream

    def test_truncate_one_without_carry_zeroes_recurrent_weight_gradient(self):
        # Every step starts from hidden 0.0, so w_hidden never contributes.
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS], truncate=1, carry_hidden=False)
        seq.backward([0.5, -0.5, 1.0, 0.25])
        self.assertEqual(seq.linear.grad[1], 0.0)


# ---------------------------------------------------------------------------
# TanhSequence: accumulation, zero_grad, apply_gradients, reset
# ---------------------------------------------------------------------------

class SequenceStateManagementTest(CloseMixin, unittest.TestCase):
    WEIGHT = [0.45, -0.35]
    BIAS = 0.08
    ROWS = [0.3, -0.7, 1.1]
    COEFFS = [0.5, -0.4, 0.8]

    def make_seq(self):
        return TanhSequence(Linear(list(self.WEIGHT), self.BIAS))

    def test_repeated_backward_accumulates_parameter_gradients(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        first = seq.backward(list(self.COEFFS))
        grad_once = list(seq.linear.grad)
        grad_bias_once = seq.linear.grad_bias
        second = seq.backward(list(self.COEFFS))
        # Returned input gradients are identical; parameter gradients double.
        self.assertListClose(first, second)
        self.assertListClose(seq.linear.grad, [2.0 * g for g in grad_once])
        self.assertClose(seq.linear.grad_bias, 2.0 * grad_bias_once)

    def test_zero_grad_clears_gradients_but_keeps_params_and_cache(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        seq.backward(list(self.COEFFS))
        seq.linear.zero_grad()
        self.assertListClose(seq.linear.grad, [0.0, 0.0])
        self.assertClose(seq.linear.grad_bias, 0.0)
        self.assertListClose(seq.linear.weight, self.WEIGHT)
        self.assertClose(seq.linear.bias, self.BIAS)
        # Cache survives: backward still works and re-accumulates.
        seq.backward(list(self.COEFFS))
        self.assertTrue(any(g != 0.0 for g in seq.linear.grad))

    def test_apply_gradients_updates_by_public_formula(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        seq.backward(list(self.COEFFS))
        grads = list(seq.linear.grad)
        grad_bias = seq.linear.grad_bias
        lr = 0.2
        seq.linear.apply_gradients(lr)
        expected_w = [w - lr * g for w, g in zip(self.WEIGHT, grads)]
        expected_b = self.BIAS - lr * grad_bias
        self.assertListClose(seq.linear.weight, expected_w)
        self.assertClose(seq.linear.bias, expected_b)
        # The updated parameters drive subsequent forwards.
        out = seq.forward([[x] for x in self.ROWS])
        self.assertListClose(
            out, reference_outputs(expected_w, expected_b, self.ROWS,
                                   None, False, 0.0))

    def test_backward_uses_forward_time_parameters_after_update(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        expected_input_grads = seq.backward(list(self.COEFFS))
        # Update parameters, clear gradients, backprop the recorded pass again.
        seq.linear.apply_gradients(0.5)
        seq.linear.zero_grad()
        input_grads = seq.backward(list(self.COEFFS))
        self.assertListClose(input_grads, expected_input_grads)

    def test_reset_clears_sequence_state_but_not_linear(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        seq.backward(list(self.COEFFS))
        grads = list(seq.linear.grad)
        grad_bias = seq.linear.grad_bias
        seq.reset()
        self.assertClose(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])
        # Cache invalidated...
        with self.assertRaises(RuntimeError):
            seq.backward(list(self.COEFFS))
        # ...but the Linear's parameters and accumulated gradients survive.
        self.assertListClose(seq.linear.weight, self.WEIGHT)
        self.assertClose(seq.linear.bias, self.BIAS)
        self.assertListClose(seq.linear.grad, grads)
        self.assertClose(seq.linear.grad_bias, grad_bias)
        # The Linear's own forward record also survives.
        lin_grads = seq.linear.backward(1.0)
        self.assertEqual(len(lin_grads), 2)

    def test_step_invalidates_cached_forward(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        seq.step([0.4])
        with self.assertRaises(RuntimeError):
            seq.backward(list(self.COEFFS))

    def test_new_forward_replaces_cache(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        seq.forward([[0.1], [0.2]])
        input_grads = seq.backward([1.0, 1.0])
        self.assertEqual(len(input_grads), 2)


# ---------------------------------------------------------------------------
# TanhSequence: error contract and failure atomicity
# ---------------------------------------------------------------------------

class SequenceErrorTest(CloseMixin, unittest.TestCase):
    WEIGHT = [0.45, -0.35]
    BIAS = 0.08
    ROWS = [0.3, -0.7, 1.1]

    def make_seq(self):
        return TanhSequence(Linear(list(self.WEIGHT), self.BIAS))

    def public_state(self, seq):
        lin = seq.linear
        return (seq.hidden, list(seq.outputs), list(lin.weight), lin.bias,
                list(lin.grad), lin.grad_bias,
                None if lin.last is None else list(lin.last))

    def test_constructor_requires_two_weight_linear(self):
        for bad in ("not a linear", Linear([1.0]), Linear([1.0, 2.0, 3.0])):
            with self.assertRaises(ValueError, msg=repr(bad)):
                TanhSequence(bad)
        # A rejected construction leaves the layer untouched.
        lin = Linear([1.0, 2.0, 3.0], bias=0.4)
        try:
            TanhSequence(lin)
        except ValueError:
            pass
        self.assertListClose(lin.weight, [1.0, 2.0, 3.0])
        self.assertClose(lin.bias, 0.4)
        self.assertListClose(lin.grad, [0.0, 0.0, 0.0])

    def test_forward_rejects_invalid_truncate(self):
        seq = self.make_seq()
        before = self.public_state(seq)
        for bad in (0, -1, 2.5, True, "2"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                seq.forward([[0.1]], truncate=bad)
        self.assertEqual(self.public_state(seq), before)

    def test_forward_rejects_non_bool_carry_hidden(self):
        seq = self.make_seq()
        before = self.public_state(seq)
        for bad in (0, 1, "yes", None, 0.0):
            with self.assertRaises(ValueError, msg=repr(bad)):
                seq.forward([[0.1]], carry_hidden=bad)
        self.assertEqual(self.public_state(seq), before)

    def test_forward_rejects_invalid_initial_hidden(self):
        seq = self.make_seq()
        before = self.public_state(seq)
        for bad in ("x", True, [0.1]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                seq.forward([[0.1]], initial_hidden=bad)
        self.assertEqual(self.public_state(seq), before)

    def test_failed_forward_leaves_no_partial_state(self):
        seq = self.make_seq()
        # Establish a successful cached forward and accumulated gradients.
        seq.forward([[x] for x in self.ROWS])
        seq.backward([0.5, -0.5, 1.0])
        before = self.public_state(seq)
        before_outputs = seq.outputs

        bad_rows = [[0.1], [0.2], ["oops"], [0.4]]
        with self.assertRaises(ValueError):
            seq.forward(bad_rows)
        # Nothing observable changed: hidden, outputs (same list object),
        # parameters, accumulated gradients and the Linear's forward record.
        after = self.public_state(seq)
        self.assertEqual(after, before)
        self.assertIs(seq.outputs, before_outputs)
        # The earlier cache is still back-propagatable.
        input_grads = seq.backward([0.5, -0.5, 1.0])
        self.assertEqual(len(input_grads), 3)

    def test_forward_rejects_bad_rows_without_prior_cache(self):
        seq = self.make_seq()
        for bad_rows in ([[1.0, 2.0]], [[True]], [["x"]], [[0.1], [0.2, 0.3]],
                         ["ab"], [[0.1], "ab"]):
            with self.assertRaises(ValueError, msg=repr(bad_rows)):
                seq.forward(bad_rows)
        self.assertEqual(seq.outputs, [])
        self.assertClose(seq.hidden, 0.0)
        with self.assertRaises(RuntimeError):
            seq.backward([1.0])

    def test_step_rejects_bad_row_and_preserves_state(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])  # establishes cache + hidden
        before = self.public_state(seq)
        before_outputs = list(seq.outputs)
        for bad_row in ([1.0, 2.0], [True], ["x"], "ab", 1.5):
            with self.assertRaises(ValueError, msg=repr(bad_row)):
                seq.step(bad_row)
        self.assertEqual(self.public_state(seq), before)
        self.assertEqual(seq.outputs, before_outputs)
        # The batch cache survives the rejected step.
        input_grads = seq.backward([1.0, 1.0, 1.0])
        self.assertEqual(len(input_grads), 3)

    def test_backward_requires_cached_forward(self):
        seq = self.make_seq()
        with self.assertRaises(RuntimeError):
            seq.backward([1.0])
        with self.assertRaises(RuntimeError):
            seq.backward_with_initial_hidden([1.0], 0.0)
        seq.forward([[0.1], [0.2]])
        seq.reset()
        with self.assertRaises(RuntimeError):
            seq.backward([1.0, 1.0])

    def test_backward_rejects_bad_grad_outputs_and_stays_reusable(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        seq.backward([0.5, -0.5, 1.0])
        grads_before = list(seq.linear.grad)
        grad_bias_before = seq.linear.grad_bias
        for bad in ([1.0], [1.0, 2.0, 3.0, 4.0], [1.0, "x", 1.0],
                    [1.0, True, 1.0], "abc"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                seq.backward(bad)
        # Failed calls accumulated nothing and the cache is intact.
        self.assertListClose(seq.linear.grad, grads_before)
        self.assertClose(seq.linear.grad_bias, grad_bias_before)
        input_grads = seq.backward([0.5, -0.5, 1.0])
        self.assertEqual(len(input_grads), 3)
        self.assertListClose(seq.linear.grad,
                             [2.0 * g for g in grads_before])

    def test_backward_with_initial_hidden_rejects_bad_grad_hidden(self):
        seq = self.make_seq()
        seq.forward([[x] for x in self.ROWS])
        for bad in ("x", True, None, [0.1]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                seq.backward_with_initial_hidden([1.0, 1.0, 1.0], bad)
        self.assertListClose(seq.linear.grad, [0.0, 0.0])
        self.assertClose(seq.linear.grad_bias, 0.0)

    def test_backward_with_initial_hidden_returns_pair(self):
        seq = self.make_seq()
        seq.forward([[0.2], [0.4]])
        result = seq.backward_with_initial_hidden([1.0, 1.0], 0.3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        input_grads, grad_init = result
        self.assertEqual(len(input_grads), 2)
        self.assertIsInstance(grad_init, float)


if __name__ == '__main__':
    unittest.main()
