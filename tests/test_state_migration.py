"""Cross-instance state migration via export_state()/import_state()."""
import copy
import json
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


def fresh_target():
    # A same-width instance with deliberately different parameters and state.
    seq = TanhSequence(Linear([9.0, -9.0], 9.0))
    seq.forward([[0.1], [0.2]])
    seq.backward([1.0, 1.0])
    return seq


def json_round_trip(state):
    return json.loads(json.dumps(state))


class ExportShapeTest(unittest.TestCase):
    def test_export_is_plain_json_compatible_data(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2,
                    segment_starts=[False, True, False, False, False])
        state = seq.export_state()
        # JSON round trip preserves the value exactly: only dicts, lists,
        # finite numbers, booleans, strings and None are present.
        self.assertEqual(json_round_trip(state), state)
        self.assertEqual(state["version"], 2)
        self.assertIsInstance(state["version"], int)
        self.assertNotIsInstance(state["version"], bool)
        self.assertEqual(state["width"], 2)
        self.assertEqual(state["forward"]["boundaries"], [0, 1, 2, 4])
        self.assertIsNone(state["stream"])

    def test_export_does_not_mutate_and_shares_no_references(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.forward(ROWS, truncate=2)
        seq.backward(GO)
        before = seq.export_state()
        again = seq.export_state()
        self.assertEqual(before, again)  # export is repeatable
        # Mutating every mutable corner of one export leaves the instance
        # and the other export untouched.
        before["outputs"].append(99.0)
        before["forward"]["inputs"][0].append(99.0)
        before["forward"]["boundaries"].append(99)
        before["forward"]["weights"][0] = 99.0
        before["linear"]["weight"][0] = 99.0
        before["linear"]["grad"][0] = 99.0
        before["linear"]["last"][0] = 99.0
        self.assertEqual(seq.export_state(), again)

    def test_export_without_cache_or_stream_uses_none(self):
        seq = TanhSequence(Linear(list(W), B))
        state = seq.export_state()
        self.assertIsNone(state["forward"])
        self.assertIsNone(state["stream"])
        self.assertIsNone(state["linear"]["last"])
        self.assertIsNone(state["linear"]["last_weight"])
        self.assertEqual(state["outputs"], [])
        self.assertEqual(state["hidden"], 0.0)


class MigrationContinuationTest(unittest.TestCase):
    def test_batch_cache_migration_continues_identically(self):
        src = TanhSequence(Linear(list(W), B))
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        src.backward_with_initial_hidden(GO, 0.1)  # accumulate param grads
        state = src.export_state()

        dst = fresh_target()
        self.assertIsNone(dst.import_state(json_round_trip(state)))
        self.assertEqual(dst.export_state(), src.export_state())

        # Every backward entry keeps agreeing value for value.
        self.assertEqual(dst.backward(GO), src.backward(GO))
        self.assertEqual(dst.backward_with_initial_hidden(GO, 0.1),
                         src.backward_with_initial_hidden(GO, 0.1))
        self.assertEqual(dst.backward_with_boundaries(GO, 0.1),
                         src.backward_with_boundaries(GO, 0.1))
        # Accumulated parameter gradients migrated and keep accumulating.
        self.assertEqual(dst.linear.grad, src.linear.grad)
        self.assertEqual(dst.linear.grad_bias, src.linear.grad_bias)
        # Stepping forward past the cached pass stays in lockstep.
        self.assertEqual(dst.step([0.6]), src.step([0.6]))
        self.assertEqual(dst.export_state(), src.export_state())

    def test_open_stream_migration_continues_identically(self):
        src = TanhSequence(Linear(list(W), B))
        src.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        src.step(ROWS[0])
        src.step(ROWS[1])
        src.step(ROWS[2], segment_start=True)
        state = src.export_state()
        self.assertEqual(state["stream"]["initial_hidden"], 0.25)
        self.assertEqual(state["stream"]["boundaries"], [0, 2])
        self.assertIsNone(state["forward"])

        dst = fresh_target()
        dst.import_state(copy.deepcopy(state))
        self.assertEqual(dst.export_state(), src.export_state())
        # More steps (one crossing a truncate boundary), then finish and
        # back-propagate through the committed trajectory on both sides.
        self.assertEqual(dst.step(ROWS[3]), src.step(ROWS[3]))
        self.assertEqual(dst.step(ROWS[4]), src.step(ROWS[4]))
        self.assertEqual(dst.finish_stream(), src.finish_stream())
        grads = [0.2, -0.1, 0.4, 0.7, -0.3]
        self.assertEqual(dst.backward_with_boundaries(grads, 0.05),
                         src.backward_with_boundaries(grads, 0.05))
        self.assertEqual(dst.export_state(), src.export_state())

    def test_empty_state_migration_keeps_empty(self):
        src = TanhSequence(Linear(list(W), B))
        dst = fresh_target()
        dst.import_state(src.export_state())
        self.assertEqual(dst.export_state(), src.export_state())
        # No cached pass on either side: backward still raises RuntimeError.
        self.assertRaises(RuntimeError, dst.backward, [1.0])
        self.assertRaises(RuntimeError, dst.backward_with_initial_hidden,
                          [1.0], 0.0)
        self.assertRaises(RuntimeError, dst.backward_with_boundaries,
                          [1.0], 0.0)
        self.assertRaises(RuntimeError, dst.finish_stream)

    def test_import_does_not_mutate_the_state_object(self):
        src = TanhSequence(Linear(list(W), B))
        src.forward(ROWS, truncate=2)
        state = src.export_state()
        pristine = copy.deepcopy(state)
        fresh_target().import_state(state)
        self.assertEqual(state, pristine)

    def test_repeated_import_is_stable(self):
        src = TanhSequence(Linear(list(W), B))
        src.forward(ROWS, truncate=2, carry_hidden=True)
        src.backward(GO)
        state = src.export_state()
        dst = fresh_target()
        dst.import_state(state)
        first = dst.export_state()
        dst.import_state(state)
        dst.import_state(copy.deepcopy(state))
        self.assertEqual(dst.export_state(), first)
        self.assertEqual(dst.export_state(), src.export_state())

    def test_migrated_instances_do_not_share_references(self):
        src = TanhSequence(Linear(list(W), B))
        src.forward(ROWS, truncate=2)
        state = src.export_state()
        dst = fresh_target()
        dst.import_state(state)
        # Diverging the target must not reach the source or the state object.
        dst.step([9.9])
        dst.linear.apply_gradients(0.5)
        self.assertEqual(src.export_state(), state)
        # ...and vice versa.
        src_before = dst.export_state()
        src.step([9.9])
        self.assertEqual(dst.export_state(), src_before)

    def test_multi_feature_width_migrates(self):
        src = TanhSequence(Linear([0.4, -0.3, 0.2], 0.1))
        src.forward([[0.5, 0.1], [-0.2, 0.7], [0.9, -0.4]], truncate=2)
        src.backward([0.1, 0.3, 0.5])
        dst = TanhSequence(Linear([1.0, 1.0, 1.0], 1.0))
        dst.import_state(json_round_trip(src.export_state()))
        self.assertEqual(dst.export_state(), src.export_state())
        grads = [0.6, 0.4, 0.2]
        self.assertEqual(dst.backward_with_initial_hidden(grads, 0.2),
                         src.backward_with_initial_hidden(grads, 0.2))


class ImportValidationTest(unittest.TestCase):
    def setUp(self):
        self.src = TanhSequence(Linear(list(W), B))
        self.src.forward(ROWS, truncate=2, carry_hidden=True,
                         initial_hidden=0.2)
        self.src.backward(GO)

    def valid_state(self):
        return self.src.export_state()

    def assert_import_rejected(self, state):
        dst = fresh_target()
        snap = dst.export_state()
        expect_value_error(lambda: dst.import_state(state))
        # A rejected import leaves the target completely unchanged.
        self.assertEqual(dst.export_state(), snap)

    def test_rejects_non_dict_and_foreign_objects(self):
        for bad in (None, 42, "state", [1, 2], self.src.checkpoint(),
                    self.src, Linear(list(W))):
            self.assert_import_rejected(bad)

    def test_rejects_missing_and_extra_top_level_keys(self):
        state = self.valid_state()
        del state["hidden"]
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["extra"] = None
        self.assert_import_rejected(state)

    def test_rejects_wrong_version(self):
        for bad in (0, 3, -1, "1", 1.0, True, None):
            state = self.valid_state()
            state["version"] = bad
            self.assert_import_rejected(state)

    def test_accepts_version_1_with_missing_seeds_as_none(self):
        # A version-1 state has no "segment_hiddens" fields; it imports
        # cleanly and every boundary value is interpreted as None.
        state = json_round_trip(self.valid_state())
        state["version"] = 1
        del state["forward"]["segment_hiddens"]
        dst = fresh_target()
        dst.import_state(state)
        # The instance re-exports in the current version with explicit
        # all-None seeds, and continues exactly like the source.
        self.assertEqual(dst.export_state(), self.src.export_state())
        self.assertEqual(dst.backward(GO), self.src.backward(GO))

    def test_rejects_wrong_kind_and_width(self):
        state = self.valid_state()
        state["kind"] = "SomethingElse"
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["kind"] = 1
        self.assert_import_rejected(state)
        for bad in (3, 1, 0, -2, "2", 2.0, True):
            state = self.valid_state()
            state["width"] = bad
            self.assert_import_rejected(state)

    def test_rejects_width_incompatible_instance(self):
        state = self.valid_state()
        wide = TanhSequence(Linear([0.1, 0.2, 0.3], 0.0))
        expect_value_error(lambda: wide.import_state(state))

    def test_rejects_nonfinite_numbers(self):
        for bad in (NAN, INF, -INF, BIG):
            state = self.valid_state()
            state["hidden"] = bad
            self.assert_import_rejected(state)
            state = self.valid_state()
            state["outputs"][1] = bad
            self.assert_import_rejected(state)
            state = self.valid_state()
            state["linear"]["weight"][0] = bad
            self.assert_import_rejected(state)
            state = self.valid_state()
            state["linear"]["grad_bias"] = bad
            self.assert_import_rejected(state)
            state = self.valid_state()
            state["forward"]["prev_hiddens"][0] = bad
            self.assert_import_rejected(state)
            state = self.valid_state()
            state["forward"]["inputs"][2][0] = bad
            self.assert_import_rejected(state)

    def test_rejects_inconsistent_trajectory(self):
        # prev_hiddens must be the values the steps actually consumed.
        state = self.valid_state()
        state["forward"]["prev_hiddens"][1] += 0.5
        self.assert_import_rejected(state)
        # Trajectory lengths must agree.
        state = self.valid_state()
        state["forward"]["outputs"] = state["forward"]["outputs"][:-1]
        self.assert_import_rejected(state)
        # Cached outputs must equal the visible outputs.
        state = self.valid_state()
        state["outputs"] = list(state["outputs"])[:-1]
        self.assert_import_rejected(state)
        # The truncate rule's cuts must all be marked as boundaries.
        state = self.valid_state()
        state["forward"]["boundaries"] = [0]
        self.assert_import_rejected(state)
        # Boundary indices must stay inside the trajectory.
        state = self.valid_state()
        state["forward"]["boundaries"] = [0, 2, 4, 99]
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["forward"]["boundaries"] = [0, 2, 4, -1]
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["forward"]["boundaries"] = [0, 2, 4, 1.5]
        self.assert_import_rejected(state)
        # Hidden must be the last produced output.
        state = self.valid_state()
        state["hidden"] = 0.12345
        self.assert_import_rejected(state)

    def test_rejects_forward_and_stream_together(self):
        streaming = TanhSequence(Linear(list(W), B))
        streaming.start_stream(initial_hidden=0.1, truncate=2)
        streaming.step(ROWS[0])
        state = self.valid_state()
        state["stream"] = streaming.export_state()["stream"]
        self.assert_import_rejected(state)

    def test_rejects_bad_linear_cache_shape(self):
        state = self.valid_state()
        state["linear"]["last"] = [1.0]  # wrong length
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["linear"]["last"] = [1.0, 2.0]
        state["linear"]["last_weight"] = None  # cache without its weights
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["linear"]["last"] = None
        state["linear"]["last_weight"] = list(W)  # weights without a cache
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["linear"]["weight"] = [0.1, 0.2, 0.3]  # width mismatch
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["linear"]["grad"] = []
        self.assert_import_rejected(state)

    def test_rejects_structurally_bad_records(self):
        state = self.valid_state()
        state["forward"]["truncate"] = 0  # must be a positive int or None
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["forward"]["truncate"] = True
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["forward"]["carry_hidden"] = 1  # must be a bool
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["forward"]["boundaries"] = {0, 2, 4}  # must be a list
        self.assert_import_rejected(state)
        state = self.valid_state()
        del state["forward"]["weights"]
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["stream"] = {"inputs": []}  # missing keys
        self.assert_import_rejected(state)
        state = self.valid_state()
        state["linear"] = list(state["linear"].values())
        self.assert_import_rejected(state)

    def test_rejected_import_preserves_open_stream(self):
        dst = TanhSequence(Linear(list(W), B))
        dst.start_stream(initial_hidden=0.1, truncate=2)
        dst.step(ROWS[0])
        snap = dst.export_state()
        state = self.valid_state()
        state["hidden"] = NAN
        expect_value_error(lambda: dst.import_state(state))
        self.assertEqual(dst.export_state(), snap)
        # The open session is still usable afterwards.
        dst.step(ROWS[1])
        self.assertEqual(len(dst.finish_stream()), 2)

    def test_stream_state_round_trip_validation(self):
        # A stream record whose step-0 prev_hidden disagrees with the
        # recorded initial_hidden violates the truncation rules.
        src = TanhSequence(Linear(list(W), B))
        src.start_stream(initial_hidden=0.25, truncate=2)
        src.step(ROWS[0])
        src.step(ROWS[1])
        state = src.export_state()
        state["stream"]["initial_hidden"] = 0.5
        self.assert_import_rejected(state)
        # ...while the untouched state imports cleanly.
        fresh_target().import_state(src.export_state())


class ExistingInterfaceTest(unittest.TestCase):
    def test_checkpoint_restore_stay_instance_bound(self):
        src = TanhSequence(Linear(list(W), B))
        src.forward(ROWS, truncate=2)
        cp = src.checkpoint()
        other = TanhSequence(Linear(list(W), B))
        expect_value_error(lambda: other.restore(cp))
        # Same-instance restore still works and returns None.
        src.step([9.9])
        self.assertIsNone(src.restore(cp))
        self.assertEqual(src.outputs, src.export_state()["outputs"])

    def test_export_import_do_not_replace_checkpoint_restore(self):
        seq = TanhSequence(Linear(list(W), B))
        seq.forward(ROWS, truncate=2)
        cp = seq.checkpoint()
        state = seq.export_state()
        seq.step([9.9])
        seq.restore(cp)
        self.assertEqual(seq.export_state(), state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
