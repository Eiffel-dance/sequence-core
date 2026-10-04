"""Tests for the export_state/import_state cross-instance state transfer."""
import copy
import json
import math
import unittest

from app import Linear, TanhSequence

W, B = [0.4, -0.3], 0.15
WM, BM = [0.4, -0.3, 0.2], 0.15
XS = [0.8, -0.5, 1.2, -0.7, 0.3]
ROWS = [[x] for x in XS]
ROWSM = [[0.8, -0.2], [-0.5, 1.1], [1.2, 0.4], [-0.7, -0.9], [0.3, 0.6]]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]
GOM = [0.3, -0.6, 0.9, -0.2, 0.5]
GH = 0.35
H0 = 0.25

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400


def fresh_sequence(weight=W, bias=B):
    return TanhSequence(Linear(list(weight), bias))


def observable(seq):
    return (seq.hidden, list(seq.outputs), list(seq.linear.weight),
            seq.linear.bias, list(seq.linear.grad), seq.linear.grad_bias,
            None if seq.linear.last is None else list(seq.linear.last))


class ExportShapeTest(unittest.TestCase):
    def test_export_is_plain_json_data(self):
        seq = fresh_sequence()
        seq.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        seq.backward(GO)
        state = seq.export_state()
        # A JSON round trip must be lossless for the whole object.
        self.assertEqual(json.loads(json.dumps(state)), state)
        self.assertEqual(state["version"], 1)
        self.assertIsInstance(state["version"], int)
        self.assertNotIsInstance(state["version"], bool)
        self.assertEqual(state["width"], 2)
        # Boundaries are an ordered integer list.
        self.assertEqual(state["forward"]["boundaries"], [0, 2, 4])
        self.assertIsNone(state["stream"])

    def test_export_does_not_mutate_and_shares_no_references(self):
        seq = fresh_sequence()
        seq.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        for x in XS[:3]:
            seq.step([x])
        before = observable(seq)
        stream_len = len(seq._stream["outputs"])
        first = seq.export_state()
        second = seq.export_state()
        self.assertEqual(observable(seq), before)
        self.assertEqual(len(seq._stream["outputs"]), stream_len)
        self.assertEqual(first, second)
        # Mutating one export must not reach the instance or the other export.
        first["outputs"].append(9.0)
        first["stream"]["inputs"][0][0] = 9.0
        first["stream"]["boundaries"].append(99)
        first["linear"]["weight"][0] = 9.0
        first["linear"]["grad"].append(9.0)
        self.assertEqual(observable(seq), before)
        self.assertNotEqual(first, second)
        self.assertEqual(second["stream"]["boundaries"], [0, 2])

    def test_empty_state_exports_nones(self):
        seq = fresh_sequence()
        state = seq.export_state()
        self.assertIsNone(state["forward"])
        self.assertIsNone(state["stream"])
        self.assertIsNone(state["linear"]["last"])
        self.assertIsNone(state["linear"]["last_weight"])
        self.assertEqual(state["outputs"], [])
        self.assertEqual(state["hidden"], 0.0)


class TransferContinuationTest(unittest.TestCase):
    def assert_same_continuation(self, source, target, rows, go):
        """Both instances continue identically from the exported moment."""
        self.assertEqual(observable(source), observable(target))
        src_outs = [source.step(list(row)) for row in rows]
        tgt_outs = [target.step(list(row)) for row in rows]
        self.assertEqual(src_outs, tgt_outs)
        self.assertEqual(observable(source), observable(target))
        self.assertEqual(source.finish_stream(), target.finish_stream())
        src = source.backward_with_boundaries(go, GH)
        tgt = target.backward_with_boundaries(go, GH)
        self.assertEqual(src, tgt)
        self.assertEqual(observable(source), observable(target))

    def test_batch_cache_transfer_then_continue(self):
        source = fresh_sequence()
        source.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        source.backward(GO)  # accumulate one pass of parameter gradients
        state = source.export_state()

        target = fresh_sequence([9.9, 9.9], -3.0)  # different live state
        target.forward(ROWS[:1])
        self.assertIsNone(target.import_state(state))
        # The batch cache is back-propagatable with identical results.
        self.assertEqual(source.backward_with_boundaries(GO, GH),
                         target.backward_with_boundaries(GO, GH))
        self.assertEqual(observable(source), observable(target))
        # Stepwise continuation matches value for value.
        self.assertEqual(source.step([0.2]), target.step([0.2]))
        self.assertEqual(observable(source), observable(target))

    def test_open_stream_transfer_then_finish(self):
        source = fresh_sequence()
        source.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        for x in XS[:3]:
            source.step([x])
        state = source.export_state()

        target = fresh_sequence()
        target.import_state(state)
        # The open session resumes on the other instance.
        self.assert_same_continuation(
            source, target, [[XS[3]], [XS[4]]], GO)

    def test_json_roundtrip_transfer(self):
        source = fresh_sequence()
        source.forward(ROWS, truncate=2, initial_hidden=H0)
        source.backward(GO)
        state = json.loads(json.dumps(source.export_state()))
        target = fresh_sequence()
        target.import_state(state)
        self.assertEqual(observable(source), observable(target))
        self.assertEqual(source.backward_with_initial_hidden(GO, GH),
                         target.backward_with_initial_hidden(GO, GH))

    def test_deep_copy_transfer(self):
        source = fresh_sequence()
        source.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=H0)
        state = copy.deepcopy(source.export_state())
        target = fresh_sequence()
        target.import_state(state)
        self.assertEqual(observable(source), observable(target))

    def test_empty_and_cacheless_states_transfer(self):
        # No cache, no stream: the target ends up equally empty.
        source = fresh_sequence()
        source.linear.forward([0.5, 0.5])  # Linear cache only
        state = source.export_state()
        target = fresh_sequence()
        target.forward(ROWS, truncate=2)
        target.import_state(state)
        self.assertEqual(observable(source), observable(target))
        self.assertIsNone(target._fwd)
        self.assertIsNone(target._stream)
        with self.assertRaises(RuntimeError):
            target.backward(GO)
        # The Linear's own last-forward record came along.
        self.assertEqual(target.linear.backward(0.5),
                         source.linear.backward(0.5))

        # An empty committed batch pass transfers as an empty cache.
        source2 = fresh_sequence()
        source2.forward([], truncate=2, initial_hidden=H0)
        target2 = fresh_sequence()
        target2.import_state(source2.export_state())
        self.assertEqual(
            source2.backward_with_initial_hidden([], GH),
            target2.backward_with_initial_hidden([], GH))

    def test_multi_feature_transfer(self):
        source = TanhSequence(Linear(list(WM), BM))
        source.forward(ROWSM, truncate=2, carry_hidden=True, initial_hidden=H0)
        source.backward(GOM)
        target = TanhSequence(Linear([0.0, 0.0, 0.0], 1.0))
        target.import_state(source.export_state())
        self.assertEqual(observable(source), observable(target))
        self.assertEqual(source.backward_with_boundaries(GOM, GH),
                         target.backward_with_boundaries(GOM, GH))

    def test_repeated_import_is_stable(self):
        source = fresh_sequence()
        source.forward(ROWS, truncate=2, initial_hidden=H0)
        source.backward(GO)
        state = source.export_state()
        target = fresh_sequence()
        target.import_state(state)
        first = observable(target)
        # Drift the live state, then re-import the same object.
        target.forward(ROWS[:1])
        target.linear.zero_grad()
        target.import_state(state)
        self.assertEqual(observable(target), first)
        target.import_state(state)
        self.assertEqual(observable(target), first)
        # The state object itself was not consumed or mutated.
        self.assertEqual(state, source.export_state())

    def test_import_does_not_alias_state_object(self):
        source = fresh_sequence()
        source.forward(ROWS, truncate=2, initial_hidden=H0)
        state = source.export_state()
        target = fresh_sequence()
        target.import_state(state)
        # Later mutation of the state object must not reach the target.
        state["outputs"][0] = 123.0
        state["linear"]["weight"][0] = 123.0
        state["forward"]["inputs"][0][0] = 123.0
        self.assertNotEqual(target.outputs[0], 123.0)
        self.assertNotEqual(target.linear.weight[0], 123.0)
        # And the two instances evolve independently after the transfer.
        source.linear.weight[0] = -7.0
        self.assertNotEqual(target.linear.weight[0], -7.0)


class ImportValidationTest(unittest.TestCase):
    def setUp(self):
        self.source = fresh_sequence()
        self.source.forward(ROWS, truncate=2, carry_hidden=True,
                            initial_hidden=H0)
        self.source.backward(GO)
        self.target = fresh_sequence()
        self.target.forward(ROWS[:2], truncate=1)
        self.target.backward(GO[:2])
        self.snapshot = observable(self.target) + (
            list(self.target._fwd["outputs"]),)

    def assert_rejected(self, state):
        with self.assertRaises(ValueError):
            self.target.import_state(state)
        current = observable(self.target) + (
            list(self.target._fwd["outputs"]),)
        self.assertEqual(current, self.snapshot)

    def valid_state(self):
        return self.source.export_state()

    def test_rejects_non_dict_and_bad_structure(self):
        for bad in (None, 7, "state", [1, 2], object()):
            self.assert_rejected(bad)
        state = self.valid_state()
        del state["hidden"]
        self.assert_rejected(state)
        state = self.valid_state()
        state["extra"] = 1
        self.assert_rejected(state)
        state = self.valid_state()
        state["linear"]["unexpected"] = 1
        self.assert_rejected(state)
        state = self.valid_state()
        del state["forward"]["weights"]
        self.assert_rejected(state)

    def test_rejects_bad_version(self):
        for bad in (2, 0, -1, "1", 1.0, True, None):
            state = self.valid_state()
            state["version"] = bad
            self.assert_rejected(state)

    def test_rejects_width_mismatch(self):
        state = self.valid_state()
        state["width"] = 3
        self.assert_rejected(state)
        # A genuine width-3 export does not fit a width-2 target either.
        wide = TanhSequence(Linear(list(WM), BM))
        wide.forward(ROWSM, truncate=2)
        self.assert_rejected(wide.export_state())
        for bad in (1, 0, "2", 2.0, True, None):
            state = self.valid_state()
            state["width"] = bad
            self.assert_rejected(state)

    def test_rejects_nonfinite_numbers(self):
        for bad in (NAN, INF, -INF, BIG):
            state = self.valid_state()
            state["hidden"] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["outputs"][1] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["linear"]["weight"][0] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["linear"]["grad"][1] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["linear"]["grad_bias"] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["forward"]["prev_hiddens"][0] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["forward"]["inputs"][2][0] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["forward"]["weights"][1] = bad
            self.assert_rejected(state)
            state = self.valid_state()
            state["linear"]["last"][0] = bad
            self.assert_rejected(state)

    def test_rejects_trajectory_and_boundary_inconsistencies(self):
        # Trajectory lengths disagree.
        state = self.valid_state()
        state["forward"]["prev_hiddens"] = state["forward"]["prev_hiddens"][:-1]
        self.assert_rejected(state)
        # Visible outputs disagree with the cached pass.
        state = self.valid_state()
        state["outputs"] = state["outputs"][:-1]
        self.assert_rejected(state)
        # Boundary index out of range / not an int.
        state = self.valid_state()
        state["forward"]["boundaries"] = [0, 2, 5]
        self.assert_rejected(state)
        state = self.valid_state()
        state["forward"]["boundaries"] = [0, 2.0, 4]
        self.assert_rejected(state)
        state = self.valid_state()
        state["forward"]["boundaries"] = {0, 2, 4}  # must be a list
        self.assert_rejected(state)
        # A truncate-required boundary is missing.
        state = self.valid_state()
        state["forward"]["boundaries"] = [0, 4]
        self.assert_rejected(state)
        # prev_hiddens violates the truncation rule at a boundary.
        state = self.valid_state()
        state["forward"]["prev_hiddens"][2] = 0.123
        self.assert_rejected(state)
        # Bad truncate / carry_hidden values.
        state = self.valid_state()
        state["forward"]["truncate"] = 0
        self.assert_rejected(state)
        state = self.valid_state()
        state["forward"]["carry_hidden"] = 1
        self.assert_rejected(state)

    def test_rejects_stream_and_cache_coexisting(self):
        streamed = fresh_sequence()
        streamed.start_stream(initial_hidden=H0, truncate=2)
        streamed.step([XS[0]])
        state = self.valid_state()
        state["stream"] = streamed.export_state()["stream"]
        self.assert_rejected(state)

    def test_rejects_hidden_state_inconsistencies(self):
        state = self.valid_state()
        state["hidden"] = 0.123  # != last output
        self.assert_rejected(state)
        # An empty open session must sit at its initial_hidden.
        streamed = fresh_sequence()
        streamed.start_stream(initial_hidden=H0, truncate=2)
        state = streamed.export_state()
        state["hidden"] = 0.0
        self.assert_rejected(state)
        # A stream's first prev_hidden must be its initial_hidden.
        streamed.step([XS[0]])
        state = streamed.export_state()
        state["stream"]["prev_hiddens"][0] = 0.0
        self.assert_rejected(state)

    def test_rejects_linear_cache_shape_errors(self):
        state = self.valid_state()
        state["linear"]["weight"] = [0.4]
        self.assert_rejected(state)
        state = self.valid_state()
        state["linear"]["grad"] = [0.0, 0.0, 0.0]
        self.assert_rejected(state)
        state = self.valid_state()
        state["linear"]["last"] = None  # last_weight still present
        self.assert_rejected(state)
        state = self.valid_state()
        state["linear"]["last_weight"] = None
        self.assert_rejected(state)
        state = self.valid_state()
        state["linear"]["last"] = [0.1]
        self.assert_rejected(state)
        state = self.valid_state()
        state["linear"]["bias"] = True
        self.assert_rejected(state)

    def test_open_stream_survives_rejected_import(self):
        target = fresh_sequence()
        target.start_stream(initial_hidden=H0, truncate=2, carry_hidden=True)
        target.step([XS[0]])
        before = (target.hidden, list(target.outputs),
                  len(target._stream["outputs"]))
        state = self.valid_state()
        state["hidden"] = NAN
        with self.assertRaises(ValueError):
            target.import_state(state)
        self.assertEqual((target.hidden, list(target.outputs),
                          len(target._stream["outputs"])), before)
        # The session is still alive and finishes normally.
        self.assertEqual(len(target.finish_stream()), 1)

    def test_successful_import_returns_none_and_restores_none_slots(self):
        # A source with no cache and no stream clears the target's cache.
        empty = fresh_sequence()
        empty.linear.forward([0.1, 0.2])
        target = fresh_sequence()
        target.forward(ROWS, truncate=2)
        self.assertIsNone(target.import_state(empty.export_state()))
        self.assertIsNone(target._fwd)
        self.assertIsNone(target._stream)
        self.assertIsNotNone(target.linear.last)


class ExistingBehaviorUnchangedTest(unittest.TestCase):
    def test_checkpoint_restore_still_instance_bound(self):
        seq = fresh_sequence()
        seq.forward(ROWS, truncate=2, initial_hidden=H0)
        cp = seq.checkpoint()
        other = fresh_sequence()
        with self.assertRaises(ValueError):
            other.restore(cp)
        seq.forward(ROWS[:1])
        self.assertIsNone(seq.restore(cp))
        self.assertEqual(seq.outputs,
                         fresh_sequence_restored_outputs())

    def test_checkpoint_and_export_are_independent(self):
        seq = fresh_sequence()
        seq.forward(ROWS, truncate=2, initial_hidden=H0)
        cp = seq.checkpoint()
        state = seq.export_state()
        seq.forward(ROWS[:2])
        seq.restore(cp)
        self.assertEqual(seq.export_state(), state)


def fresh_sequence_restored_outputs():
    seq = fresh_sequence()
    seq.forward(ROWS, truncate=2, initial_hidden=H0)
    return list(seq.outputs)


if __name__ == "__main__":
    unittest.main()
