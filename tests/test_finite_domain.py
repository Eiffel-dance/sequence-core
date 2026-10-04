"""Finite-domain validation probes (not part of the shipped suite)."""
import math
import unittest

from app import Linear, TanhSequence

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400  # finite int, but overflows double arithmetic

W, B = [0.4, -0.3], 0.15
ROWS = [[0.8], [-0.5], [1.2], [-0.7], [0.3]]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("ValueError not raised by %r" % fn)


class LinearFiniteDomainTest(unittest.TestCase):
    def test_construction_rejects_nonfinite(self):
        for bad in ([NAN], [INF], [-INF], [BIG], [1.0, NAN]):
            expect_value_error(lambda bad=bad: Linear(bad))
        expect_value_error(lambda: Linear([1.0], NAN))
        expect_value_error(lambda: Linear([1.0], INF))
        expect_value_error(lambda: Linear([BIG], 0.0))

    def test_forward_rejects_nonfinite_inputs_without_touching_cache(self):
        lin = Linear([1.0, 2.0], 0.5)
        lin.forward([3.0, 4.0])
        cached_last, cached_w = list(lin.last), list(lin._last_weight)
        for bad in ([NAN, 0.0], [INF, 0.0], [0.0, BIG], [10 ** 309, 1.0]):
            expect_value_error(lambda bad=bad: lin.forward(bad))
        self.assertEqual(lin.last, cached_last)
        self.assertEqual(lin._last_weight, cached_w)

    def test_forward_rejects_overflowing_result(self):
        lin = Linear([1e308])
        self.assertIsNone(lin.last)
        expect_value_error(lambda: lin.forward([10.0]))  # 1e309 -> inf
        self.assertIsNone(lin.last)
        # Huge int triggers OverflowError inside the arithmetic, also ValueError.
        expect_value_error(lambda: lin.forward([BIG]))
        self.assertIsNone(lin.last)
        # Retry with a finite, in-range value succeeds and caches normally.
        self.assertTrue(math.isfinite(lin.forward([0.1])))
        self.assertEqual(lin.last, [0.1])

    def test_backward_rejects_nonfinite_upstream(self):
        lin = Linear(list(W), B)
        lin.forward([0.8, -0.5])
        lin.backward(0.6)
        g, gb, last = list(lin.grad), lin.grad_bias, list(lin.last)
        for bad in (NAN, INF, -INF, BIG):
            expect_value_error(lambda bad=bad: lin.backward(bad))
        self.assertEqual(lin.grad, g)
        self.assertEqual(lin.grad_bias, gb)
        self.assertEqual(lin.last, last)
        # Retry succeeds and accumulates exactly one more valid pass.
        lin.backward(0.25)
        self.assertTrue(all(abs(a - (b + 0.25 * c)) < 1e-12
                            for a, b, c in zip(lin.grad, g, [0.8, -0.5])))

    def test_backward_rejects_overflowing_result_without_partial_write(self):
        lin = Linear([1e200, 1e200])
        lin.forward([1.0, 1.0])  # 2e200 is finite
        g_before, gb_before = list(lin.grad), lin.grad_bias
        expect_value_error(lambda: lin.backward(1e200))  # grad*w -> inf
        self.assertEqual(lin.grad, g_before)
        self.assertEqual(lin.grad_bias, gb_before)
        # Retry with a finite upstream works and yields the published formula.
        out = lin.backward(1.0)
        self.assertTrue(all(math.isfinite(v) for v in out))
        self.assertEqual(out, [1e200, 1e200])

    def test_apply_gradients_rejects_nonfinite_lr(self):
        lin = Linear(list(W), B)
        lin.forward([1.0, 0.0])
        for bad in (NAN, INF, -INF, BIG):
            expect_value_error(lambda bad=bad: lin.apply_gradients(bad))
        self.assertEqual(lin.weight, W)
        self.assertEqual(lin.bias, B)

    def test_apply_gradients_rejects_overflowing_update_atomically(self):
        lin = Linear([1e308, 1.0], 0.0)
        lin.grad = [-1e308, 0.0]
        w_before, b_before = list(lin.weight), lin.bias
        # w - lr*g = 1e308 + 10*1e308 -> inf for the first weight.
        expect_value_error(lambda: lin.apply_gradients(10.0))
        self.assertEqual(lin.weight, w_before)
        self.assertEqual(lin.bias, b_before)
        # A finite retry updates exactly by the published formula.
        lin.apply_gradients(0.1)
        self.assertTrue(math.isfinite(lin.weight[0]))
        self.assertAlmostEqual(lin.weight[0], 1e308 - 0.1 * (-1e308), delta=1e294)


class SequenceFiniteDomainTest(unittest.TestCase):
    def setUp(self):
        self.seq = TanhSequence(Linear(list(W), B))
        self.seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.25)
        self.seq.backward(GO)

    def snapshot(self):
        return (self.seq.hidden, list(self.seq.outputs),
                list(self.seq.linear.last), list(self.seq.linear.weight),
                list(self.seq.linear.grad), self.seq.linear.grad_bias)

    def test_forward_rejects_nonfinite_initial_hidden(self):
        snap = self.snapshot()
        for bad in (NAN, INF, -INF, BIG):
            expect_value_error(
                lambda bad=bad: self.seq.forward(ROWS, initial_hidden=bad))
            self.assertEqual(self.snapshot(), snap)

    def test_forward_validates_every_row_before_state_change(self):
        snap = self.snapshot()
        # Bad value at the very end must not leave the earlier rows traversed.
        for bad_rows in ([[NAN]], [[INF]], [[1.0], [NAN]],
                         [[1.0], [BIG]], [[1.0], [10 ** 309]]):
            expect_value_error(
                lambda bad_rows=bad_rows: self.seq.forward(bad_rows, truncate=2))
            self.assertEqual(self.snapshot(), snap)
        # Linear forward record is restored too, so the old cache backprops.
        self.assertEqual(len(self.seq.backward(GO)), len(ROWS))
        # A valid retry of the same entry works normally.
        outs = self.seq.forward(ROWS, truncate=2, initial_hidden=0.25)
        self.assertEqual(len(outs), len(ROWS))
        self.assertTrue(all(math.isfinite(v) for v in outs))

    def test_forward_rejects_overflowing_preactivation(self):
        seq = TanhSequence(Linear([1e308, 1.0], 0.0))
        expect_value_error(lambda: seq.forward([[10.0]]))  # inf before tanh
        self.assertIsNone(seq.linear.last)
        self.assertEqual(seq.hidden, 0.0)
        self.assertEqual(seq.outputs, [])

    def test_step_rejects_nonfinite_row_and_retries(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.forward(ROWS[:2])
        h, outs, last = seq.hidden, list(seq.outputs), list(seq.linear.last)
        for bad in ([NAN], [INF], [BIG]):
            expect_value_error(lambda bad=bad: seq.step(bad))
        self.assertEqual((seq.hidden, list(seq.outputs), list(seq.linear.last)),
                         (h, outs, last))
        good = seq.step([0.2])
        self.assertTrue(math.isfinite(good))

    def test_start_stream_rejects_nonfinite_seed(self):
        snap = self.snapshot()
        for bad in (NAN, INF, -INF, BIG):
            expect_value_error(
                lambda bad=bad: self.seq.start_stream(initial_hidden=bad))
        self.assertEqual(self.snapshot(), snap)
        # No session was opened by the rejected calls.
        with self.assertRaises(RuntimeError):
            self.seq.finish_stream()
        # Valid start works and a fresh session behaves normally.
        self.seq.start_stream(initial_hidden=0.25, truncate=2)
        self.seq.step([ROWS[0][0]])
        self.assertEqual(len(self.seq.finish_stream()), 1)

    def test_bad_stream_step_only_rejects_that_step(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        seq.step([ROWS[0][0]])
        h0, outs0, last0 = seq.hidden, list(seq.outputs), list(seq.linear.last)
        for bad in ([NAN], [INF], [BIG], []):
            expect_value_error(lambda bad=bad: seq.step(bad))
        self.assertEqual((seq.hidden, list(seq.outputs), list(seq.linear.last)),
                         (h0, outs0, last0))
        # Session still alive; fixing the row continues the full trajectory.
        for row in ROWS[1:]:
            seq.step(row)
        returned = seq.finish_stream()
        self.assertEqual(len(returned), len(ROWS))
        self.assertTrue(all(math.isfinite(v) for v in returned))
        # Backprop on that session still matches a finite batch pass.
        ref = TanhSequence(Linear(list(W), B))
        ref.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.25)
        self.assertTrue(all(
            abs(a - b) <= 1e-12
            for a, b in zip(seq.backward(GO), ref.backward(GO))))

    def test_backward_entries_validate_full_list_and_seed_first(self):
        snap_grad = (list(self.seq.linear.grad), self.seq.linear.grad_bias)
        bad_lists = [GO[:i] + [NAN] + GO[i + 1:] for i in range(len(GO))]
        bad_lists += [GO[:i] + [INF] + GO[i + 1:] for i in range(len(GO))]
        bad_lists += [[BIG if j == 0 else v for j, v in enumerate(GO)]]
        for bad in bad_lists:
            expect_value_error(lambda bad=bad: self.seq.backward(bad))
            expect_value_error(
                lambda bad=bad: self.seq.backward_with_initial_hidden(bad, 0.35))
            expect_value_error(
                lambda bad=bad: self.seq.backward_with_boundaries(bad, 0.35))
        for bad_seed in (NAN, INF, BIG):
            expect_value_error(
                lambda s=bad_seed: self.seq.backward_with_initial_hidden(GO, s))
            expect_value_error(
                lambda s=bad_seed: self.seq.backward_with_boundaries(GO, s))
        # Nothing accumulated; cache still backpropagates; a valid retry works.
        self.assertEqual((list(self.seq.linear.grad), self.seq.linear.grad_bias),
                         snap_grad)
        ig, gh = self.seq.backward_with_initial_hidden(GO, 0.35)
        self.assertEqual(len(ig), len(GO))
        self.assertTrue(math.isfinite(gh))

    def test_backward_overflow_accumulates_nothing(self):
        # prev_hidden = 1e200 with w_h = 1e-200 gives a moderate pre-activation
        # (nonzero tanh derivative), but a 1e308 upstream makes the hidden
        # parameter gradient d_pre * 1e200 overflow during backprop.
        seq = TanhSequence(Linear([1e-200, 1e-200], 0.0))
        seq.forward([[0.0]], initial_hidden=1e200)
        before = (list(seq.linear.grad), seq.linear.grad_bias)
        expect_value_error(lambda: seq.backward([1e308]))
        self.assertEqual((list(seq.linear.grad), seq.linear.grad_bias), before)
        # Finite retry succeeds.
        ig = seq.backward([0.1])
        self.assertTrue(math.isfinite(ig[0]))

    def test_empty_sequence_rejects_nonfinite_seed(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.forward([], truncate=2, initial_hidden=0.25)
        for bad in (NAN, INF, BIG):
            expect_value_error(
                lambda s=bad: seq.backward_with_initial_hidden([], s))
        ig, gh = seq.backward_with_initial_hidden([], 0.5)
        self.assertEqual(ig, [])
        self.assertEqual(gh, 0.5)

    def test_multi_feature_nonfinite_row_rolls_back(self):
        wm = [0.4, -0.3, 0.2]
        seq = TanhSequence(Linear(list(wm), B))
        rows = [[0.8, -0.2], [-0.5, 1.1], [1.2, 0.4]]
        seq.forward(rows, truncate=2)
        snap = (seq.hidden, list(seq.outputs), list(seq.linear.last))
        for bad in ([[NAN, 0.0], [0.0, 0.0]], [[0.0, BIG], [0.0, 0.0]]):
            expect_value_error(lambda bad=bad: seq.forward(bad, truncate=2))
            self.assertEqual(
                (seq.hidden, list(seq.outputs), list(seq.linear.last)), snap)
        out = seq.forward(rows, truncate=2)
        self.assertEqual(len(out), 3)
        gs = seq.backward([0.1, 0.2, 0.3])
        self.assertEqual([len(r) for r in gs], [2, 2, 2])


class CheckpointFiniteDomainTest(unittest.TestCase):
    def setUp(self):
        self.seq = TanhSequence(Linear(list(W), B))
        self.seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.25)
        self.seq.backward(GO)

    def snapshot(self):
        return (self.seq.hidden, list(self.seq.outputs),
                list(self.seq.linear.weight), self.seq.linear.bias,
                list(self.seq.linear.grad), self.seq.linear.grad_bias,
                list(self.seq.linear.last))

    def assert_restore_rejected(self, cp):
        snap = self.snapshot()
        expect_value_error(lambda: self.seq.restore(cp))
        self.assertEqual(self.snapshot(), snap)

    def test_restore_rejects_nonfinite_scalar_slots(self):
        cp = self.seq.checkpoint()
        cp._hidden = NAN
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._linear_bias = INF
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._linear_grad_bias = BIG  # overflow-sized int
        self.assert_restore_rejected(cp)

    def test_restore_rejects_nonfinite_lists_and_records(self):
        cp = self.seq.checkpoint()
        cp._linear_weight[0] = NAN
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._linear_grad[1] = INF
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._outputs[0] = BIG
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._fwd["outputs"][1] = NAN
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._fwd["prev_hiddens"][0] = INF
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._fwd["inputs"][0][0] = NAN
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._fwd["weights"][0] = BIG
        self.assert_restore_rejected(cp)
        cp = self.seq.checkpoint()
        cp._linear_last[0] = NAN
        self.assert_restore_rejected(cp)

    def test_restore_rejects_nonfinite_open_stream(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        seq.step(ROWS[0])
        cp = seq.checkpoint()
        cp._stream["initial_hidden"] = NAN
        snap = (seq.hidden, list(seq.outputs), len(seq._stream["outputs"]))
        expect_value_error(lambda: seq.restore(cp))
        # Open session must survive the rejected restore untouched.
        self.assertEqual((seq.hidden, list(seq.outputs),
                          len(seq._stream["outputs"])), snap)
        self.assertEqual(len(seq.finish_stream()), 1)

    def test_valid_checkpoint_restores_repeatedly(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.forward(ROWS, truncate=2, initial_hidden=0.25)
        cp = seq.checkpoint()
        outs_first = list(seq.outputs)
        seq.forward([[9.0], [9.0]])
        seq.backward([1.0, 1.0])
        for _ in range(3):
            seq.restore(cp)
            self.assertEqual(list(seq.outputs), outs_first)
            self.assertTrue(seq._fwd is not None)
        # Repeated restore keeps backprop working with identical results.
        a = seq.backward(GO)
        seq.restore(cp)
        b = seq.backward(GO)
        self.assertTrue(all(abs(x - y) <= 1e-15 for x, y in zip(a, b)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
