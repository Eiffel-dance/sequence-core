"""Targeted probes for stream-step failure atomicity (task spec)."""
import copy
import math
import unittest

from app import Linear, TanhSequence

BIGW = 1e308  # * 10 -> inf inside a finite, declared-legal row -> ValueError


def full_snapshot(seq):
    """Deep snapshot of every observable datum a failed step must not touch."""
    s = seq._stream
    return {
        "hidden": seq.hidden,
        "outputs": list(seq.outputs),
        "fwd": copy.deepcopy(seq._fwd),
        "stream": None if s is None else {
            "inputs": [list(r) for r in s["inputs"]],
            "prev_hiddens": list(s["prev_hiddens"]),
            "outputs": list(s["outputs"]),
            "boundaries": set(s["boundaries"]),
            "truncate": s["truncate"],
            "carry_hidden": s["carry_hidden"],
            "initial_hidden": s["initial_hidden"],
            "weights": list(s["weights"]),
        },
        "weight": list(seq.linear.weight),
        "bias": seq.linear.bias,
        "grad": list(seq.linear.grad),
        "grad_bias": seq.linear.grad_bias,
        "last": None if seq.linear.last is None else list(seq.linear.last),
        "last_weight": None if seq.linear._last_weight is None
        else list(seq.linear._last_weight),
    }


class OverflowAtomicityTest(unittest.TestCase):
    def _probe(self, d, truncate, carry, h0, steps, fail_index, fail_kw,
               valid_rows):
        # d=1 scalar weights [BIGW, 0.5]; d=2 [BIGW, 0.3, 0.5]
        weights = [BIGW] + [0.3] * (d - 1) + [0.5]
        seq = TanhSequence(Linear(weights, 0.1))
        seq.start_stream(initial_hidden=h0, truncate=truncate,
                         carry_hidden=carry)
        for row, kw in steps:
            seq.step(row, **kw)
        before = full_snapshot(seq)

        # A finite, declared-legal row whose LINEAR arithmetic overflows.
        overflow_row = [10.0] * d
        with self.assertRaises(ValueError):
            seq.step(overflow_row, **fail_kw)
        after = full_snapshot(seq)
        self.assertEqual(after, before,
                         "failed stream step leaked state: %r vs %r"
                         % (after, before))

        # Immediate retry of the identical call fails identically and still
        # leaves no trace.
        with self.assertRaises(ValueError):
            seq.step(overflow_row, **fail_kw)
        self.assertEqual(full_snapshot(seq), before)

        # Complete the session with valid rows; boundaries, outputs and every
        # backward entry must match a twin that never saw a failure.
        go = [0.2 * (t + 1) for t in range(fail_index + len(valid_rows))]
        for row in valid_rows:
            seq.step(list(row))
        returned = seq.finish_stream()
        self.assertEqual(len(returned), fail_index + len(valid_rows))

        twin = TanhSequence(Linear(weights, 0.1))
        twin.start_stream(initial_hidden=h0, truncate=truncate,
                          carry_hidden=carry)
        for row, kw in steps:
            twin.step(row, **kw)
        for row in valid_rows:
            twin.step(list(row))
        expected = twin.finish_stream()

        self.assertEqual(returned, expected)
        self.assertEqual(seq._fwd["boundaries"], twin._fwd["boundaries"])
        self.assertEqual(seq._fwd["prev_hiddens"], twin._fwd["prev_hiddens"])
        seq.backward(go)
        twin.backward(go)
        self.assertEqual(seq.linear.grad, twin.linear.grad)
        self.assertEqual(seq.linear.grad_bias, twin.linear.grad_bias)
        return seq, twin, go

    def test_overflow_at_truncate_boundary_scalar(self):
        # Two successful steps, then index 2 is a truncate=2 boundary.
        self._probe(1, 2, False, 0.25,
                    [([0.1], {}), ([0.2], {})], 2, {},
                    [[0.3], [-0.4]])

    def test_overflow_at_carry_boundary_scalar(self):
        self._probe(1, 2, True, 0.25,
                    [([0.1], {}), ([0.2], {})], 2, {},
                    [[0.3], [-0.4], [0.05]])

    def test_overflow_at_explicit_segment_start(self):
        # Explicit mark at a non-truncate index: the old code leaked the mark.
        self._probe(1, None, False, 0.0,
                    [([0.1], {}), ([0.2], {})], 2,
                    {"segment_start": True}, [[0.3], [-0.4]])

    def test_overflow_at_explicit_start_carry_multifeature(self):
        self._probe(2, 3, True, 0.2,
                    [([0.1, -0.2], {}), ([0.2, 0.1], {})], 2,
                    {"segment_start": True}, [[0.3, 0.0], [-0.4, 0.2]])

    def test_overflow_on_first_step_empty_session(self):
        # No prior steps: boundary 0 would have been leaked by the old code.
        seq = TanhSequence(Linear([BIGW, 0.5], 0.1))
        seq.start_stream(initial_hidden=0.25, truncate=2)
        before = full_snapshot(seq)
        with self.assertRaises(ValueError):
            seq.step([10.0])
        self.assertEqual(full_snapshot(seq), before)
        with self.assertRaises(ValueError):
            seq.step([10.0], segment_start=True)
        self.assertEqual(full_snapshot(seq), before)
        # Valid retry reproduces a never-failed first step exactly.
        out = seq.step([0.3])
        twin = TanhSequence(Linear([BIGW, 0.5], 0.1))
        twin.start_stream(initial_hidden=0.25, truncate=2)
        self.assertEqual(out, twin.step([0.3]))
        seq.finish_stream()
        twin.finish_stream()
        self.assertEqual(seq._fwd["boundaries"], twin._fwd["boundaries"])
        gi1 = seq.backward_with_initial_hidden([0.7], 0.4)
        gi2 = twin.backward_with_initial_hidden([0.7], 0.4)
        self.assertEqual(gi1, gi2)

    def test_overflow_keeps_last_successful_linear_cache_value_for_value(self):
        seq = TanhSequence(Linear([BIGW, 0.5], 0.1))
        seq.start_stream(initial_hidden=0.25)
        seq.step([0.3])  # 1e308*0.3 is finite (3e307); records the cache
        cached_last = list(seq.linear.last)
        cached_w = list(seq.linear._last_weight)
        with self.assertRaises(ValueError):
            seq.step([10.0])
        self.assertEqual(seq.linear.last, cached_last)
        self.assertEqual(seq.linear._last_weight, cached_w)
        # Stream trajectory entry 0 matches the retained cache.
        self.assertEqual(seq._stream["inputs"], [[0.3]])
        self.assertEqual(len(seq._stream["prev_hiddens"]), 1)

    def test_checkpoint_and_export_after_failure_match_clean_twin(self):
        weights = [BIGW, 0.5]
        seq = TanhSequence(Linear(weights, 0.1))
        seq.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        seq.step([0.1])
        seq.step([0.2])
        with self.assertRaises(ValueError):
            seq.step([10.0])  # overflow at boundary index 2

        twin = TanhSequence(Linear(weights, 0.1))
        twin.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        twin.step([0.1])
        twin.step([0.2])

        # Exported open-session state must be that of the clean twin...
        self.assertEqual(seq.export_state(), twin.export_state())
        cp = seq.checkpoint()
        # ...and it must restore cleanly (the old leaked-boundary record was
        # semantically inconsistent and would be rejected).
        seq.step([0.3])
        seq.restore(cp)
        self.assertEqual(seq.export_state(), twin.export_state())
        # Finish both and compare backward.
        seq.step([0.3])
        outs = seq.finish_stream()
        twin.step([0.3])
        expected = twin.finish_stream()
        self.assertEqual(outs, expected)
        ig1, gh1, bg1 = seq.backward_with_boundaries([0.2, 0.4, 0.6], 0.3)
        ig2, gh2, bg2 = twin.backward_with_boundaries([0.2, 0.4, 0.6], 0.3)
        self.assertEqual((ig1, gh1, bg1), (ig2, gh2, bg2))

    def test_non_session_step_overflow_is_atomic(self):
        seq = TanhSequence(Linear([BIGW, 0.5], 0.1))
        seq.forward([[0.1], [0.2]], initial_hidden=0.25)
        before = full_snapshot(seq)
        with self.assertRaises(ValueError):
            seq.step([10.0])
        self.assertEqual(full_snapshot(seq), before)

    def test_validation_errors_still_stateless(self):
        seq = TanhSequence(Linear([0.4, -0.3], 0.1))
        seq.start_stream(truncate=2)
        seq.step([0.1])
        before = full_snapshot(seq)
        for bad_row in ([], [1.0, 2.0], ["x"], [True], 7, "ab",
                        [float("nan")], [10 ** 400]):
            with self.assertRaises(ValueError):
                seq.step(bad_row)
        with self.assertRaises(ValueError):
            seq.step([0.2], segment_start=1)
        self.assertEqual(full_snapshot(seq), before)
        seq.finish_stream()

    def test_segment_start_without_session_remains_runtime_error(self):
        fresh = TanhSequence(Linear([0.4, -0.3], 0.1))
        with self.assertRaises(RuntimeError):
            fresh.step([0.2], segment_start=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
