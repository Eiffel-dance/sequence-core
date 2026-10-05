"""Configurable segment-boundary hidden states (segment_hiddens)."""
import copy
import json
import math
import unittest

from app import (ELUSequence, GELUSequence, LeakyReLUSequence, Linear,
                 ReLUSequence, SigmoidSequence, SiLUSequence,
                 SoftplusSequence, TanhSequence)

INF = float("inf")
NAN = float("nan")
BIG = 10 ** 400

W, B = [0.4, -0.3], 0.15
ROWS = [[0.8], [-0.5], [1.2], [-0.7], [0.3]]
GO = [0.3, -0.6, 0.9, -0.2, 0.5]

ALL_CLASSES = (TanhSequence, SigmoidSequence, SoftplusSequence, ReLUSequence,
               LeakyReLUSequence, ELUSequence, GELUSequence, SiLUSequence)


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("ValueError not raised by %r" % fn)


def expect_runtime_error(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("RuntimeError not raised by %r" % fn)


def fresh(cls=TanhSequence):
    return cls(Linear(list(W), B))


def json_round_trip(state):
    return json.loads(json.dumps(state))


def reference_outputs(rows, h0, seeds, boundaries, carry):
    outs = []
    hidden = h0
    for i, row in enumerate(rows):
        if i in boundaries:
            seed = seeds.get(i)
            if seed is not None:
                hidden = seed
            elif i == 0:
                hidden = h0
            elif carry:
                hidden = outs[-1]
            else:
                hidden = 0.0
        hidden = math.tanh(W[0] * row[0] + W[1] * hidden + B)
        outs.append(hidden)
    return outs


class BatchForwardTest(unittest.TestCase):
    def test_seed_overrides_carry_and_zero_reset(self):
        # Boundaries at 0 (truncate), 2 (truncate) and 3 (declared); seeds
        # at 2 and 3 replace both the carried value and the zero reset.
        for carry in (False, True):
            seq = fresh()
            got = seq.forward(
                ROWS, truncate=2, carry_hidden=carry, initial_hidden=0.2,
                segment_starts=[False, False, False, True, False],
                segment_hiddens=[None, None, 0.7, -0.4, None])
            expected = reference_outputs(
                ROWS, 0.2, {2: 0.7, 3: -0.4}, {0, 2, 3, 4}, carry)
            self.assertEqual(got, expected)
            self.assertEqual(seq.outputs, expected)
            self.assertEqual(seq.hidden, expected[-1])

    def test_seed_at_position_zero_overrides_initial_hidden(self):
        seq = fresh()
        got = seq.forward(ROWS, initial_hidden=0.9, segment_hiddens=[-0.6, None, None, None, None])
        # Position 0 is always a boundary; the seed replaces initial_hidden.
        self.assertEqual(seq.export_state()["forward"]["segment_hiddens"],
                         [-0.6, None, None, None, None])
        self.assertEqual(seq.export_state()["forward"]["boundaries"], [0])
        expected = reference_outputs(ROWS, -0.6, {0: -0.6}, {0}, False)
        self.assertEqual(got, expected)

    def test_all_none_segment_hiddens_matches_baseline(self):
        base = fresh()
        base_out = base.forward(ROWS, truncate=2, carry_hidden=True,
                                initial_hidden=0.2)
        seq = fresh()
        got = seq.forward(ROWS, truncate=2, carry_hidden=True,
                          initial_hidden=0.2,
                          segment_hiddens=[None] * len(ROWS))
        self.assertEqual(got, base_out)
        self.assertEqual(seq.export_state(), base.export_state())

    def test_omitted_segment_hiddens_keeps_baseline(self):
        base = fresh()
        base_out = base.forward(ROWS, truncate=2, carry_hidden=True)
        seq = fresh()
        got = seq.forward(ROWS, truncate=2, carry_hidden=True)
        self.assertEqual(got, base_out)
        self.assertEqual(seq.export_state(), base.export_state())

    def test_int_seeds_accepted(self):
        seq = fresh()
        got = seq.forward(ROWS, truncate=2, segment_hiddens=[1, None, -2, None, None])
        expected = reference_outputs(ROWS, 1, {0: 1, 2: -2}, {0, 2, 4}, False)
        # Position 4 is a truncate boundary without a seed: zero reset.
        self.assertEqual(got, expected)

    def test_validation_rejects_bad_segment_hiddens(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2)
        snap = seq.export_state()
        linear_snap = (list(seq.linear.weight), seq.linear.bias,
                       list(seq.linear.grad), seq.linear.grad_bias,
                       seq.linear.last)
        bad_calls = [
            # Length and container type.
            lambda: seq.forward(ROWS, segment_hiddens=[None] * 4),
            lambda: seq.forward(ROWS, segment_hiddens=[None] * 6),
            lambda: seq.forward(ROWS, segment_hiddens="none"),
            lambda: seq.forward(ROWS, segment_hiddens=42),
            lambda: seq.forward(ROWS, segment_hiddens={0: 1.0}),
            # Element types and domain.
            lambda: seq.forward(ROWS, segment_hiddens=[True] + [None] * 4),
            lambda: seq.forward(ROWS, segment_hiddens=["0.5"] + [None] * 4),
            lambda: seq.forward(ROWS, segment_hiddens=[NAN] + [None] * 4),
            lambda: seq.forward(ROWS, segment_hiddens=[INF] + [None] * 4),
            lambda: seq.forward(ROWS, segment_hiddens=[-INF] + [None] * 4),
            lambda: seq.forward(ROWS, segment_hiddens=[BIG] + [None] * 4),
            # Values where no boundary exists (no truncate, no declared
            # start past position 0).
            lambda: seq.forward(ROWS, segment_hiddens=[None, 0.5] + [None] * 3),
            lambda: seq.forward(ROWS, segment_hiddens=[None] * 4 + [0.5]),
            lambda: seq.forward(ROWS, truncate=3,
                                segment_hiddens=[None, None, None, None, 0.5]),
        ]
        for call in bad_calls:
            expect_value_error(call)
            # A rejected call touches nothing.
            self.assertEqual(seq.export_state(), snap)
            self.assertEqual((list(seq.linear.weight), seq.linear.bias,
                              list(seq.linear.grad), seq.linear.grad_bias,
                              seq.linear.last), linear_snap)

    def test_seed_positions_allowed_by_truncate_or_declared_starts(self):
        seq = fresh()
        # Boundaries: 0 and 3 from truncate, 2 from the declared start.
        got = seq.forward(ROWS, truncate=3,
                          segment_starts=[False, False, True, False, False],
                          segment_hiddens=[None, None, 0.1, 0.2, None])
        expected = reference_outputs(ROWS, 0.0, {2: 0.1, 3: 0.2},
                                     {0, 2, 3}, False)
        self.assertEqual(got, expected)


class StreamStepTest(unittest.TestCase):
    def test_batch_and_stream_agree(self):
        seeds = [None, None, 0.7, -0.4, None]
        flags = [False, False, False, True, False]
        batch = fresh()
        batch_out = batch.forward(
            ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2,
            segment_starts=flags, segment_hiddens=seeds)
        stream = fresh()
        stream.start_stream(initial_hidden=0.2, truncate=2, carry_hidden=True)
        stepped = [stream.step(row, segment_start=flag, segment_hidden=seed)
                   for row, flag, seed in zip(ROWS, flags, seeds)]
        self.assertEqual(stepped, batch_out)
        self.assertEqual(stream.finish_stream(), batch_out)
        self.assertEqual(stream.hidden, batch.hidden)
        self.assertEqual(stream.outputs, batch.outputs)
        # Parameter gradients accumulated through the shared backward pass
        # agree item by item as well.
        self.assertEqual(stream.backward_with_boundaries(GO, 0.1),
                         batch.backward_with_boundaries(GO, 0.1))
        self.assertEqual(stream.linear.grad, batch.linear.grad)
        self.assertEqual(stream.linear.grad_bias, batch.linear.grad_bias)
        self.assertEqual(stream.export_state(), batch.export_state())

    def test_stream_seed_at_truncate_boundary_without_segment_start(self):
        batch = fresh()
        batch_out = batch.forward(ROWS, truncate=2,
                                  segment_hiddens=[None, None, 0.9, None, None])
        stream = fresh()
        stream.start_stream(truncate=2)
        stepped = []
        for i, row in enumerate(ROWS):
            stepped.append(stream.step(row, segment_hidden=0.9 if i == 2
                                       else None))
        self.assertEqual(stepped, batch_out)
        self.assertEqual(stream.finish_stream(), batch_out)

    def test_stream_seed_at_position_zero(self):
        stream = fresh()
        stream.start_stream(initial_hidden=0.8)
        first = stream.step(ROWS[0], segment_hidden=-0.3)
        batch = fresh()
        batch_first = batch.forward([ROWS[0]], initial_hidden=0.8,
                                    segment_hiddens=[-0.3])
        self.assertEqual([first], batch_first)

    def test_step_segment_hidden_validation(self):
        seq = fresh()
        seq.start_stream(truncate=2)
        seq.step(ROWS[0])
        snap = seq.export_state()
        # A value at a non-boundary step is rejected and touches nothing.
        expect_value_error(lambda: seq.step(ROWS[1], segment_hidden=0.5))
        self.assertEqual(seq.export_state(), snap)
        # Bool, non-number and non-finite seeds are ValueError everywhere.
        for bad in (True, "0.5", NAN, INF, -INF, BIG):
            expect_value_error(lambda bad=bad: seq.step(ROWS[1],
                                                        segment_hidden=bad))
            expect_value_error(lambda bad=bad: seq.step(
                ROWS[1], segment_start=True, segment_hidden=bad))
        self.assertEqual(seq.export_state(), snap)
        # The session is still usable afterwards.
        seq.step(ROWS[1])
        self.assertEqual(len(seq.finish_stream()), 2)

    def test_segment_hidden_without_stream_is_runtime_error(self):
        seq = fresh()
        expect_runtime_error(lambda: seq.step(ROWS[0], segment_hidden=0.5))
        expect_runtime_error(lambda: seq.step(ROWS[0], segment_start=True,
                                              segment_hidden=0.5))
        # A plain step still works and the failed calls changed nothing.
        seq.step(ROWS[0])
        self.assertEqual(len(seq.outputs), 1)


class BackwardTest(unittest.TestCase):
    def test_boundary_seeds_are_constants_and_cut_the_gradient(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True,
                    segment_hiddens=[None, None, 0.7, None, None])
        input_grads, grad_h0, boundary_grads = \
            seq.backward_with_boundaries(GO, 0.1)
        # The seeded boundary at 2 and the plain carried boundary at 4 are
        # reported ascending and without duplicates; index 0 is not listed.
        self.assertEqual([i for i, _ in boundary_grads], [2, 4])
        # No gradient crosses a boundary: the gradient with respect to the
        # initial hidden only collects the first segment's contribution.
        # Compare against a standalone first-segment pass.
        first_seg = fresh()
        first_seg.forward(ROWS[:2])
        _, expected_h0 = first_seg.backward_with_initial_hidden(GO[:2], 0.0)
        # The upstream gradient arriving at step 1 from later steps is cut
        # at boundary 2, so the seed gradient matches the isolated segment.
        self.assertEqual(grad_h0, expected_h0)
        # The boundary gradient at 2 is the derivative of the objective with
        # respect to the detached seed constant. The finite difference must
        # detach the same way the backward pass does: the value carried into
        # the boundary at 4 is frozen (via another seed) at its unperturbed
        # number, so the perturbation cannot leak past the boundary.
        frozen4 = seq.outputs[3]

        def objective(seed):
            probe = fresh()
            outs = probe.forward(
                ROWS, truncate=2, carry_hidden=True,
                segment_hiddens=[None, None, seed, None, frozen4])
            return sum(g * o for g, o in zip(GO, outs)) + 0.1 * outs[-1]
        eps = 1e-6
        numeric = (objective(0.7 + eps) - objective(0.7 - eps)) / (2 * eps)
        self.assertAlmostEqual(dict(boundary_grads)[2], numeric, places=6)

    def test_backward_and_backward_with_initial_hidden_shapes(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2,
                    segment_hiddens=[0.1, None, -0.2, None, None])
        flat = seq.backward(GO)
        self.assertEqual(len(flat), len(ROWS))
        # A zero terminal seed reproduces backward() bit for bit.
        nested, grad_h0 = seq.backward_with_initial_hidden(GO, 0.0)
        self.assertEqual(flat, nested)
        self.assertIsInstance(grad_h0, float)

    def test_overflowing_backward_leaves_state_untouched(self):
        seq = TanhSequence(Linear([0.0, 1e308], 0.0))
        seq.forward([[0.0], [0.0]], truncate=1,
                    segment_hiddens=[0.0, None])
        snap = seq.export_state()
        expect_value_error(lambda: seq.backward([1e308, 1e308]))
        self.assertEqual(seq.export_state(), snap)


class PersistenceTest(unittest.TestCase):
    def test_checkpoint_restore_preserves_seeds(self):
        seq = fresh()
        seq.forward(ROWS, truncate=2, carry_hidden=True,
                    segment_hiddens=[None, None, 0.7, None, None])
        cp = seq.checkpoint()
        seq.step([9.9])
        seq.restore(cp)
        twin = fresh()
        twin.forward(ROWS, truncate=2, carry_hidden=True,
                     segment_hiddens=[None, None, 0.7, None, None])
        self.assertEqual(seq.export_state(), twin.export_state())
        # Backward through the restored cache agrees value for value.
        self.assertEqual(seq.backward_with_boundaries(GO, 0.1),
                         twin.backward_with_boundaries(GO, 0.1))

    def test_checkpoint_restore_mid_stream_preserves_seeds(self):
        seq = fresh()
        seq.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        seq.step(ROWS[0])
        seq.step(ROWS[1])
        seq.step(ROWS[2], segment_hidden=0.6)
        cp = seq.checkpoint()
        seq.step(ROWS[3])
        seq.restore(cp)
        twin = fresh()
        twin.start_stream(initial_hidden=0.25, truncate=2, carry_hidden=True)
        twin.step(ROWS[0])
        twin.step(ROWS[1])
        twin.step(ROWS[2], segment_hidden=0.6)
        self.assertEqual(seq.export_state(), twin.export_state())
        # The restored session continues exactly like the untouched one.
        self.assertEqual(seq.step(ROWS[3]), twin.step(ROWS[3]))
        self.assertEqual(seq.finish_stream(), twin.finish_stream())
        self.assertEqual(seq.backward(GO[:4]), twin.backward(GO[:4]))

    def test_export_import_round_trip_with_seeds(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True,
                    segment_starts=[False, False, False, True, False],
                    segment_hiddens=[None, None, 0.7, -0.4, None])
        src.backward(GO)
        state = src.export_state()
        self.assertEqual(state["version"], 2)
        self.assertEqual(state["forward"]["segment_hiddens"],
                         [None, None, 0.7, -0.4, None])
        # The state is plain JSON-compatible data.
        self.assertEqual(json_round_trip(state), state)
        dst = fresh()
        dst.linear.apply_gradients(0.01)  # diverge the target first
        dst.import_state(json_round_trip(state))
        self.assertEqual(dst.export_state(), state)
        self.assertEqual(dst.backward_with_boundaries(GO, 0.1),
                         src.backward_with_boundaries(GO, 0.1))

    def test_export_import_open_stream_with_seeds(self):
        src = fresh()
        src.start_stream(initial_hidden=0.25, truncate=2)
        src.step(ROWS[0])
        src.step(ROWS[1])
        src.step(ROWS[2], segment_hidden=0.6)
        state = src.export_state()
        self.assertEqual(state["stream"]["segment_hiddens"],
                         [None, None, 0.6])
        dst = fresh()
        dst.import_state(json_round_trip(state))
        self.assertEqual(dst.export_state(), state)
        self.assertEqual(dst.step(ROWS[3]), src.step(ROWS[3]))
        self.assertEqual(dst.finish_stream(), src.finish_stream())

    def test_import_shares_no_references(self):
        src = fresh()
        src.forward(ROWS, truncate=2, segment_hiddens=[0.1, None, 0.2, None, None])
        state = src.export_state()
        dst = fresh()
        dst.import_state(state)
        state["forward"]["segment_hiddens"][0] = 99.0
        self.assertEqual(dst.export_state()["forward"]["segment_hiddens"],
                         [0.1, None, 0.2, None, None])

    def test_import_rejects_bad_seed_records(self):
        src = fresh()
        src.forward(ROWS, truncate=2, segment_hiddens=[None, None, 0.7, None, None])
        dst = fresh()
        dst.forward(ROWS)
        snap = dst.export_state()
        tampers = [
            # Not a list / wrong length.
            lambda s: s["forward"].update(segment_hiddens=None),
            lambda s: s["forward"].update(segment_hiddens=[None] * 4),
            lambda s: s["forward"].update(segment_hiddens="x"),
            # Bad elements.
            lambda s: s["forward"].update(
                segment_hiddens=[None, None, True, None, None]),
            lambda s: s["forward"].update(
                segment_hiddens=[None, None, NAN, None, None]),
            lambda s: s["forward"].update(
                segment_hiddens=[None, None, INF, None, None]),
            # A seed where no boundary is recorded.
            lambda s: s["forward"].update(
                segment_hiddens=[None, 0.5, None, None, None]),
            # A seed that disagrees with the recorded prev_hiddens.
            lambda s: s["forward"].update(
                segment_hiddens=[None, None, 0.8, None, None]),
        ]
        for tamper in tampers:
            state = copy.deepcopy(src.export_state())
            tamper(state)
            expect_value_error(lambda state=state: dst.import_state(state))
            self.assertEqual(dst.export_state(), snap)

    def test_version_1_state_imports_with_none_seeds(self):
        src = fresh()
        src.forward(ROWS, truncate=2, carry_hidden=True, initial_hidden=0.2)
        state = json_round_trip(src.export_state())
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
        stream_state = json_round_trip(streaming.export_state())
        stream_state["version"] = 1
        del stream_state["stream"]["segment_hiddens"]
        dst2 = fresh()
        dst2.import_state(stream_state)
        self.assertEqual(dst2.export_state(), streaming.export_state())


class AllClassesTest(unittest.TestCase):
    def test_every_sequence_class_supports_segment_hiddens(self):
        for cls in ALL_CLASSES:
            seq = cls(Linear(list(W), B))
            got = seq.forward(ROWS, truncate=2, carry_hidden=True,
                              segment_hiddens=[None, None, 0.7, None, None])
            self.assertEqual(seq.export_state()["forward"]["segment_hiddens"],
                             [None, None, 0.7, None, None])
            state = json_round_trip(seq.export_state())
            dst = cls(Linear(list(W), B))
            dst.import_state(state)
            self.assertEqual(dst.export_state(), state)
            self.assertEqual(dst.backward(GO), seq.backward(GO))
            # Stream path agrees with the batch path.
            stream = cls(Linear(list(W), B))
            stream.start_stream(truncate=2, carry_hidden=True)
            stepped = [stream.step(row, segment_hidden=0.7 if i == 2 else None)
                       for i, row in enumerate(ROWS)]
            self.assertEqual(stepped, got)
            self.assertEqual(stream.finish_stream(), got)

    def test_gelu_batch_stream_parity_with_seeds(self):
        batch = GELUSequence(Linear(list(W), B))
        got = batch.forward(ROWS, truncate=2,
                            segment_hiddens=[0.3, None, -0.5, None, None])
        stream = GELUSequence(Linear(list(W), B))
        stream.start_stream(truncate=2)
        stepped = [stream.step(row, segment_hidden=seed)
                   for row, seed in zip(ROWS, [0.3, None, -0.5, None, None])]
        self.assertEqual(stepped, got)
        stream.finish_stream()
        self.assertEqual(stream.backward_with_boundaries(GO, 0.1),
                         batch.backward_with_boundaries(GO, 0.1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
