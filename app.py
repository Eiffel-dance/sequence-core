import math
import weakref
from collections.abc import Sequence


def _is_finite(value):
    """Return whether a non-bool int/float lies in the finite numeric domain.

    NaN, positive/negative infinity and ints too large to convert to a finite
    double (whose later use in float arithmetic would raise OverflowError) are
    all treated as non-finite.
    """
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _require_finite(value, what):
    """Reject NaN, infinities and arithmetic-overflow-sized ints."""
    if not _is_finite(value):
        raise ValueError("%s must be a finite number" % what)
    return value


def _read_number(value, what):
    """Validate a scalar argument; bool is not accepted as a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a Python int or float" % what)
    _require_finite(value, what)
    return value


def _read_number_sequence(values, expected_length, what):
    """Validate a non-text, non-bytes sequence of int/float values."""
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ValueError("%s must be a non-text, non-bytes sequence of numbers" % what)
    if not isinstance(values, Sequence):
        raise ValueError("%s must be a sequence of numbers" % what)
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            "%s length %d does not match weight length %d"
            % (what, len(values), expected_length)
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("each element of %s must be an int or float" % what)
        _require_finite(value, "each element of %s" % what)
        result.append(value)
    return result


def _read_row(row, width):
    """Validate a non-text sequence of ``width`` numbers and return them."""
    if isinstance(row, (str, bytes, bytearray, memoryview)):
        raise ValueError("each input row must be a non-text numeric sequence")
    try:
        length = len(row)
    except TypeError:
        raise ValueError("each input row must be a sequence of numbers")
    if length != width:
        raise ValueError(
            "each input row must contain exactly %d numeric value(s)" % width)
    values = []
    for value in row:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("each input value must be an int or float")
        _require_finite(value, "each input value")
        values.append(value)
    return values


def _read_bool_sequence(values, what):
    """Validate a non-text, non-bytes sequence of booleans and return it."""
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ValueError("%s must be a non-text sequence of booleans" % what)
    if not isinstance(values, Sequence):
        raise ValueError("%s must be a sequence of booleans" % what)
    result = []
    for value in values:
        # Only an actual bool is accepted (bool is a subclass of int).
        if not isinstance(value, bool):
            raise ValueError("each element of %s must be a bool" % what)
        result.append(value)
    return result


def _read_grad_list(values, expected_length):
    """Validate an output-gradient list and return it as a list of floats."""
    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise ValueError("grad_outputs must be a list of numbers matching the cached sequence")
    try:
        length = len(values)
    except TypeError:
        raise ValueError("grad_outputs must be a list of numbers matching the cached sequence")
    if length != expected_length:
        raise ValueError(
            "grad_outputs length %d does not match cached sequence length %d"
            % (length, expected_length)
        )
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("each output gradient must be a number")
        _require_finite(value, "each output gradient")
        result.append(float(value))
    return result


class Linear:
    def __init__(self, weight, bias=0.0):
        weights = _read_number_sequence(weight, None, "weight")
        _read_number(bias, "bias")
        self.weight = weights
        self.bias = bias
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0
        self.last = None
        # Snapshot of the weights used by the cached forward pass. Kept
        # separate from self.weight so a parameter update between forward and
        # backward cannot change the input gradients of the recorded pass.
        self._last_weight = None

    def forward(self, x):
        values = _read_number_sequence(x, len(self.weight), "x")
        # Compute before caching anything. Overflow (e.g. an int+float sum
        # beyond the double range) and a non-finite result are both rejected
        # before the cache is touched.
        try:
            result = sum(w * a for w, a in zip(self.weight, values)) + self.bias
        except OverflowError:
            raise ValueError("linear output must be finite")
        _require_finite(result, "linear output")
        # Cache only after validation and computation succeed. Record the
        # inputs together with the exact weights that produced the output.
        self.last = values
        self._last_weight = list(self.weight)
        return result

    def backward(self, grad):
        if self.last is None:
            raise RuntimeError("backward requires a successful forward call first")
        _read_number(grad, "grad")
        # Build every result into locals and verify finiteness before touching
        # the accumulated parameter gradients, so a call whose arithmetic
        # overflows leaves grad/grad_bias and the cache exactly as they were.
        try:
            new_grad = [g + grad * a for g, a in zip(self.grad, self.last)]
            new_grad_bias = self.grad_bias + grad
            input_grads = [grad * w for w in self._last_weight]
        except OverflowError:
            raise ValueError("backward result must be finite")
        for value in new_grad:
            _require_finite(value, "weight gradient")
        _require_finite(new_grad_bias, "bias gradient")
        for value in input_grads:
            _require_finite(value, "input gradient")
        # Gradients are accumulated item by item across repeated calls, using
        # the cached inputs of the recorded forward pass.
        self.grad = new_grad
        self.grad_bias = new_grad_bias
        # Input gradients must use the weights as they were at forward time.
        return input_grads

    def zero_grad(self):
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0

    def apply_gradients(self, learning_rate):
        _read_number(learning_rate, "learning_rate")
        # Compute the complete update locally and reject it unless every new
        # parameter is finite; only then commit, so parameters are never
        # partially written by a failed update.
        try:
            new_weight = [w - learning_rate * g
                         for w, g in zip(self.weight, self.grad)]
            new_bias = self.bias - learning_rate * self.grad_bias
        except OverflowError:
            raise ValueError("updated parameters must be finite")
        for value in new_weight:
            _require_finite(value, "updated weight")
        _require_finite(new_bias, "updated bias")
        self.weight = new_weight
        self.bias = new_bias

    def clip_gradients(self, max_norm):
        # Explicit gradient-clipping entry point for the CPU training loop:
        # grad and grad_bias are treated as one vector and limited as a whole
        # by the global L2 norm (no per-component clipping). Everything is
        # validated and computed into locals first; a rejected call leaves
        # grad/grad_bias exactly as they were, and no other state (weight,
        # bias, the forward cache, or any TanhSequence state) is ever touched.
        if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
            raise ValueError("max_norm must be a Python int or float")
        _require_finite(max_norm, "max_norm")
        if max_norm < 0:
            raise ValueError("max_norm must be non-negative")
        # The accumulated gradients are validated as a whole before any
        # arithmetic: a non-numeric or non-finite component anywhere rejects
        # the call with both gradient fields untouched.
        components = []
        for value in self.grad:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("each element of grad must be an int or float")
            _require_finite(value, "each element of grad")
            components.append(value)
        if isinstance(self.grad_bias, bool) \
                or not isinstance(self.grad_bias, (int, float)):
            raise ValueError("grad_bias must be an int or float")
        _require_finite(self.grad_bias, "grad_bias")
        components.append(self.grad_bias)
        # Global L2 norm by its mathematical definition. math.hypot evaluates
        # sqrt of the sum of squares without a squaring intermediate, so a
        # finite component whose square would overflow the double range still
        # yields the correct observable norm instead of a spurious overflow.
        norm = math.hypot(*components)
        _require_finite(norm, "gradient norm")
        # A zero norm means every component is already zero, and any norm
        # within the limit needs no scaling; both leave the gradients exactly
        # as they are.
        if norm <= max_norm:
            return 1.0
        # Scale every component by the same ratio. A zero max_norm with a
        # nonzero norm scales everything to exactly zero and reports 0.0.
        scale = max_norm / norm
        try:
            new_grad = [g * scale for g in self.grad]
            new_grad_bias = self.grad_bias * scale
        except OverflowError:
            raise ValueError("clipped gradients must be finite")
        for value in new_grad:
            _require_finite(value, "clipped weight gradient")
        _require_finite(new_grad_bias, "clipped bias gradient")
        self.grad = new_grad
        self.grad_bias = new_grad_bias
        return scale


class _Checkpoint:
    # Immutable-looking snapshot container returned by TanhSequence.checkpoint.
    # It stores only independent copies; nothing it references aliases live
    # instance state. The weakref ties the checkpoint to the exact
    # TanhSequence that created it, so another instance rejects it on restore.
    __slots__ = (
        "_owner_ref", "_width", "_hidden", "_outputs", "_fwd", "_stream",
        "_linear_weight", "_linear_bias", "_linear_grad", "_linear_grad_bias",
        "_linear_last", "_linear_last_weight",
    )

    def __init__(self, owner_ref, width, hidden, outputs, fwd, stream,
                 linear_weight, linear_bias, linear_grad, linear_grad_bias,
                 linear_last, linear_last_weight):
        self._owner_ref = owner_ref
        self._width = width
        self._hidden = hidden
        self._outputs = outputs
        self._fwd = fwd
        self._stream = stream
        self._linear_weight = linear_weight
        self._linear_bias = linear_bias
        self._linear_grad = linear_grad
        self._linear_grad_bias = linear_grad_bias
        self._linear_last = linear_last
        self._linear_last_weight = linear_last_weight


# Version of the portable state format produced by TanhSequence.export_state.
# Only version 1 exists; import_state rejects anything else.
_STATE_VERSION = 1
# Type tag carried by every exported state so a foreign dict is rejected
# before any field is interpreted.
_STATE_KIND = "TanhSequenceState"


class TanhSequence:
    def __init__(self, linear):
        # A Linear with at least two weights: the first d = len(weight) - 1
        # weights are the input-feature coefficients, the last weight is the
        # recurrent hidden-state coefficient, and the Linear bias is reused.
        # Rejected immediately, without touching the passed layer's
        # parameters, gradients, or last-forward record.
        if not isinstance(linear, Linear):
            raise ValueError("TanhSequence requires a Linear with at least two weights")
        if len(linear.weight) < 2:
            raise ValueError("TanhSequence requires a Linear with at least two weights")
        self.linear = linear
        # Number of input features per row. d == 1 keeps the original scalar
        # sequence behavior (flat gradient lists); d > 1 switches the
        # backward entries to nested per-feature rows.
        self.d = len(linear.weight) - 1
        self.hidden = 0.0
        self.outputs = []
        self._fwd = None
        # Active stream session record, or None when no session is open.
        self._stream = None

    def reset(self):
        self.hidden = 0.0
        self.outputs = []
        # Resetting invalidates any cached sequence forward pass, but leaves
        # the wrapped Linear's parameters, accumulated gradients, and its own
        # last-forward record completely untouched.
        self._fwd = None
        # Any half-finished stream session is abandoned as well.
        self._stream = None

    def checkpoint(self):
        # Captures the full state required to resume computation later and
        # returns it as a checkpoint that belongs to this instance only. The
        # call never mutates any state and may be made on a fresh object, after
        # reset(), right after a batch forward, or in the middle of a stream
        # session. Every mutable datum is deep-copied, so the checkpoint shares
        # no reference with live state and survives parameter updates,
        # zero_grad()/reset(), and continued stepping; the same checkpoint can
        # be restored any number of times.
        fwd = self._fwd
        if fwd is None:
            fwd_copy = None
        else:
            fwd_copy = {
                "inputs": [list(row) for row in fwd["inputs"]],
                "prev_hiddens": list(fwd["prev_hiddens"]),
                "outputs": list(fwd["outputs"]),
                "boundaries": set(fwd["boundaries"]),
                "truncate": fwd["truncate"],
                "carry_hidden": fwd["carry_hidden"],
                "weights": list(fwd["weights"]),
            }
        stream = self._stream
        if stream is None:
            stream_copy = None
        else:
            stream_copy = {
                "inputs": [list(row) for row in stream["inputs"]],
                "prev_hiddens": list(stream["prev_hiddens"]),
                "outputs": list(stream["outputs"]),
                "boundaries": set(stream["boundaries"]),
                "truncate": stream["truncate"],
                "carry_hidden": stream["carry_hidden"],
                "initial_hidden": stream["initial_hidden"],
                "weights": list(stream["weights"]),
            }
        return _Checkpoint(
            owner_ref=weakref.ref(self),
            width=self.d + 1,
            hidden=self.hidden,
            outputs=list(self.outputs),
            fwd=fwd_copy,
            stream=stream_copy,
            linear_weight=list(self.linear.weight),
            linear_bias=self.linear.bias,
            linear_grad=list(self.linear.grad),
            linear_grad_bias=self.linear.grad_bias,
            linear_last=None if self.linear.last is None
            else list(self.linear.last),
            linear_last_weight=None if self.linear._last_weight is None
            else list(self.linear._last_weight),
        )

    def restore(self, checkpoint):
        # Validate first, then commit atomically: a rejected checkpoint must
        # leave hidden state, outputs, any open stream session, the Linear's
        # parameters/gradients and its forward cache exactly as they were.
        # A checkpoint from another TanhSequence instance (even one wrapping
        # an equal-valued Linear) or one whose weight width disagrees with the
        # current Linear is rejected, as is any structurally corrupted one.
        if not isinstance(checkpoint, _Checkpoint):
            raise ValueError("restore requires a checkpoint returned by checkpoint()")

        def fail():
            raise ValueError("checkpoint is corrupted")

        def _validate():
            owner = checkpoint._owner_ref()
            if owner is not self:
                raise ValueError(
                    "checkpoint belongs to a different TanhSequence instance")
            if not isinstance(checkpoint._width, int) \
                    or isinstance(checkpoint._width, bool) or checkpoint._width < 2:
                fail()
            if checkpoint._width != len(self.linear.weight):
                raise ValueError(
                    "checkpoint width %d is incompatible with the current "
                    "Linear width %d"
                    % (checkpoint._width, len(self.linear.weight)))

            # A scalar is any non-bool int/float; it must also lie in the
            # finite numeric domain (no NaN, infinities or overflow-sized
            # ints), otherwise restoring it could poison later arithmetic.
            def need_number(value):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    fail()
                if not _is_finite(value):
                    raise ValueError("checkpoint contains a non-finite numeric value")

            def need_number_list(value, length=None):
                if not isinstance(value, list):
                    fail()
                if length is not None and len(value) != length:
                    fail()
                for item in value:
                    need_number(item)

            width = checkpoint._width
            need_number(checkpoint._hidden)
            need_number_list(checkpoint._outputs)
            need_number_list(checkpoint._linear_weight, width)
            need_number(checkpoint._linear_bias)
            need_number_list(checkpoint._linear_grad, width)
            need_number(checkpoint._linear_grad_bias)
            if checkpoint._linear_last is None:
                if checkpoint._linear_last_weight is not None:
                    fail()
            else:
                need_number_list(checkpoint._linear_last, width)
                if checkpoint._linear_last_weight is None:
                    fail()
                need_number_list(checkpoint._linear_last_weight, width)

            def validate_trajectory(record, extra_keys, expect_initial=False):
                # Structural validation first: exact key set, scalar/list
                # types, finite numeric domain, equal per-step lengths and
                # boundary indices inside the valid range.
                if record is None:
                    return
                if not isinstance(record, dict):
                    fail()
                keys = {"inputs", "prev_hiddens", "outputs", "boundaries",
                        "truncate", "carry_hidden", "weights"} | extra_keys
                if set(record.keys()) != keys:
                    fail()
                truncate = record["truncate"]
                if truncate is not None:
                    if (isinstance(truncate, bool)
                            or not isinstance(truncate, int) or truncate <= 0):
                        fail()
                if not isinstance(record["carry_hidden"], bool):
                    fail()
                need_number_list(record["weights"], width)
                n = len(record["outputs"])
                if (len(record["inputs"]) != n
                        or len(record["prev_hiddens"]) != n):
                    fail()
                need_number_list(record["outputs"])
                need_number_list(record["prev_hiddens"])
                if not isinstance(record["inputs"], list):
                    fail()
                for row in record["inputs"]:
                    need_number_list(row, width - 1)
                if not isinstance(record["boundaries"], set):
                    fail()
                for index in record["boundaries"]:
                    if not isinstance(index, int) or isinstance(index, bool) \
                            or not 0 <= index < n:
                        fail()
                # Open-stream records alone remember the session's starting
                # hidden state; read it only once the exact key set above has
                # been confirmed.
                initial_hidden = None
                if expect_initial:
                    initial_hidden = record["initial_hidden"]
                    need_number(initial_hidden)
                # Semantic validation: the arrays must describe one and the
                # same traversal. Every period the truncate rule cuts must be
                # present in the boundary set (explicit starts may add more),
                # and each prev_hiddens entry must be the value that step
                # actually consumed: a stream's first step consumes its
                # initial_hidden; a nonzero boundary consumes the previous
                # segment's last output when carrying, else exactly zero;
                # every other step continues from the previous output.
                boundaries = record["boundaries"]
                if truncate is not None:
                    for index in range(n):
                        if index % truncate == 0 and index not in boundaries:
                            fail()
                outputs = record["outputs"]
                prev_hiddens = record["prev_hiddens"]
                carry = record["carry_hidden"]
                for t in range(n):
                    if t == 0:
                        # Only open-stream records remember the session's
                        # initial_hidden; a committed batch record has no
                        # stored starting value to compare step 0 against.
                        if initial_hidden is not None \
                                and prev_hiddens[0] != initial_hidden:
                            fail()
                    elif t in boundaries:
                        expected = outputs[t - 1] if carry else 0.0
                        if prev_hiddens[t] != expected:
                            fail()
                    elif prev_hiddens[t] != outputs[t - 1]:
                        fail()

            # The committed batch pass and an open stream session are mutually
            # exclusive in live state, so a checkpoint carrying both could
            # never have been produced and is rejected outright.
            fwd_record = checkpoint._fwd
            stream = checkpoint._stream
            if fwd_record is not None and stream is not None:
                fail()

            # The committed batch pass. Its outputs must equal the visible
            # outputs captured at checkpoint time.
            if fwd_record is not None:
                validate_trajectory(fwd_record, set())
                if fwd_record["outputs"] != checkpoint._outputs:
                    fail()

            # The open stream session (None when no session was active).
            if stream is not None:
                validate_trajectory(stream, {"initial_hidden"},
                                    expect_initial=True)
                # An open session is always reflected in the visible outputs.
                if stream["outputs"] != checkpoint._outputs:
                    fail()

            # The public hidden state must agree with the visible trajectory:
            # it is the last produced output whenever one exists; an empty open
            # session still sits exactly at its initial_hidden. With no
            # trajectory and no open session the finite-domain check above is
            # all the hidden slot can be judged against.
            if checkpoint._outputs:
                if checkpoint._hidden != checkpoint._outputs[-1]:
                    fail()
            elif stream is not None:
                if checkpoint._hidden != stream["initial_hidden"]:
                    fail()

        # All validation happens before any state is touched, so a rejected
        # checkpoint (including one with deleted/replaced attributes) always
        # raises ValueError and leaves the instance completely unchanged.
        try:
            _validate()
        except ValueError:
            raise
        except (AttributeError, TypeError):
            raise ValueError("checkpoint is corrupted")

        # Everything validated: commit copies of the checkpoint's data so a
        # later mutation (or another restore) can never reach the live state
        # through a shared reference, and repeatedly restoring one checkpoint
        # always yields the same trajectory. An open session in the current
        # instance is simply overwritten.
        fwd_copy = None
        if checkpoint._fwd is not None:
            fwd_copy = {
                "inputs": [list(row) for row in checkpoint._fwd["inputs"]],
                "prev_hiddens": list(checkpoint._fwd["prev_hiddens"]),
                "outputs": list(checkpoint._fwd["outputs"]),
                "boundaries": set(checkpoint._fwd["boundaries"]),
                "truncate": checkpoint._fwd["truncate"],
                "carry_hidden": checkpoint._fwd["carry_hidden"],
                "weights": list(checkpoint._fwd["weights"]),
            }
        stream_copy = None
        if checkpoint._stream is not None:
            stream_copy = {
                "inputs": [list(row) for row in checkpoint._stream["inputs"]],
                "prev_hiddens": list(checkpoint._stream["prev_hiddens"]),
                "outputs": list(checkpoint._stream["outputs"]),
                "boundaries": set(checkpoint._stream["boundaries"]),
                "truncate": checkpoint._stream["truncate"],
                "carry_hidden": checkpoint._stream["carry_hidden"],
                "initial_hidden": checkpoint._stream["initial_hidden"],
                "weights": list(checkpoint._stream["weights"]),
            }

        self.hidden = checkpoint._hidden
        self.outputs = list(checkpoint._outputs)
        self._fwd = fwd_copy
        self._stream = stream_copy
        self.linear.weight = list(checkpoint._linear_weight)
        self.linear.bias = checkpoint._linear_bias
        self.linear.grad = list(checkpoint._linear_grad)
        self.linear.grad_bias = checkpoint._linear_grad_bias
        self.linear.last = None if checkpoint._linear_last is None \
            else list(checkpoint._linear_last)
        self.linear._last_weight = None \
            if checkpoint._linear_last_weight is None \
            else list(checkpoint._linear_last_weight)
        return None

    def export_state(self):
        # Portable counterpart of checkpoint(): captures exactly the same
        # state, but as an independent plain-data object (only dicts, lists,
        # finite numbers, booleans, strings and None) that is not tied to
        # this instance and can migrate to any other TanhSequence wrapping a
        # Linear of the same width. The call never mutates any state, and
        # every mutable datum is freshly built, so the result shares no
        # reference with the instance or with any other export, survives
        # later computation, and round-trips through JSON unchanged. Boundary
        # sets travel as sorted integer lists; an absent batch cache or
        # stream session is represented by None.
        def dump_trajectory(record, include_initial):
            data = {
                "inputs": [list(row) for row in record["inputs"]],
                "prev_hiddens": list(record["prev_hiddens"]),
                "outputs": list(record["outputs"]),
                "boundaries": sorted(record["boundaries"]),
                "truncate": record["truncate"],
                "carry_hidden": record["carry_hidden"],
                "weights": list(record["weights"]),
            }
            if include_initial:
                data["initial_hidden"] = record["initial_hidden"]
            return data

        fwd = self._fwd
        stream = self._stream
        return {
            "version": _STATE_VERSION,
            "kind": _STATE_KIND,
            "width": self.d + 1,
            "hidden": self.hidden,
            "outputs": list(self.outputs),
            "forward": None if fwd is None else dump_trajectory(fwd, False),
            "stream": None if stream is None else dump_trajectory(stream, True),
            "linear": {
                "weight": list(self.linear.weight),
                "bias": self.linear.bias,
                "grad": list(self.linear.grad),
                "grad_bias": self.linear.grad_bias,
                "last": None if self.linear.last is None
                else list(self.linear.last),
                "last_weight": None if self.linear._last_weight is None
                else list(self.linear._last_weight),
            },
        }

    def import_state(self, state):
        # Validate completely, then commit atomically. Structural validation
        # (plain-data types, exact field sets, version and kind) happens here;
        # the parsed state is then turned into a checkpoint owned by this
        # instance and funneled through restore(), which re-validates every
        # semantic rule (finite numeric domain, width compatibility,
        # trajectory/boundary consistency, the truncation hidden-state rules,
        # batch-cache/stream exclusivity, Linear cache shapes) before
        # committing, so any rejected state raises ValueError and leaves the
        # sequence state, the Linear's parameters, its cache and its
        # gradients exactly as they were. The input object is never mutated,
        # the committed state shares no reference with it, and importing the
        # same object repeatedly yields the same result.
        checkpoint = self._parse_exported_state(state)
        return self.restore(checkpoint)

    def _parse_exported_state(self, state):
        # Structural validation of a plain-data state object and conversion
        # into a _Checkpoint owned by this instance. Only the boundary lists
        # are converted (to the sets restore() expects); every other value is
        # passed through for restore()'s semantic validation.
        def fail():
            raise ValueError("state has an invalid structure")

        if not isinstance(state, dict):
            raise ValueError(
                "import_state requires a state produced by export_state()")
        if set(state.keys()) != {"version", "kind", "width", "hidden",
                                 "outputs", "forward", "stream", "linear"}:
            fail()
        version = state["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            fail()
        if version != _STATE_VERSION:
            raise ValueError("unsupported state version %r" % (version,))
        if state["kind"] != _STATE_KIND:
            fail()
        width = state["width"]
        if isinstance(width, bool) or not isinstance(width, int) or width < 2:
            fail()
        linear = state["linear"]
        if not isinstance(linear, dict) \
                or set(linear.keys()) != {"weight", "bias", "grad",
                                          "grad_bias", "last", "last_weight"}:
            fail()

        def parse_trajectory(record, expect_initial):
            if record is None:
                return None
            if not isinstance(record, dict):
                fail()
            keys = {"inputs", "prev_hiddens", "outputs", "boundaries",
                    "truncate", "carry_hidden", "weights"}
            if expect_initial:
                keys = keys | {"initial_hidden"}
            if set(record.keys()) != keys:
                fail()
            boundaries = record["boundaries"]
            if not isinstance(boundaries, list):
                fail()
            for index in boundaries:
                if isinstance(index, bool) or not isinstance(index, int):
                    fail()
            # Shallow copy so the caller's dict is never mutated; restore()
            # deep-copies everything it commits, so no reference to the
            # input object can reach the live state.
            parsed = dict(record)
            parsed["boundaries"] = set(boundaries)
            return parsed

        return _Checkpoint(
            owner_ref=weakref.ref(self),
            width=width,
            hidden=state["hidden"],
            outputs=state["outputs"],
            fwd=parse_trajectory(state["forward"], False),
            stream=parse_trajectory(state["stream"], True),
            linear_weight=linear["weight"],
            linear_bias=linear["bias"],
            linear_grad=linear["grad"],
            linear_grad_bias=linear["grad_bias"],
            linear_last=linear["last"],
            linear_last_weight=linear["last_weight"],
        )

    def step(self, row, segment_start=False):
        # Validate before any state is touched, so a rejected row changes
        # neither hidden/outputs nor the Linear's last-forward record, and a
        # batch forward cache stays available for backward().
        if not isinstance(segment_start, bool):
            raise ValueError("segment_start must be a boolean")
        values = _read_row(row, self.d)
        stream = self._stream
        if stream is not None:
            return self._stream_step(stream, values, segment_start)
        if segment_start:
            # Explicit boundary marks are only meaningful inside a session;
            # outside one there is no recorded trajectory to cut.
            raise RuntimeError(
                "segment_start=True requires an active stream session; "
                "call start_stream first")
        # Compute into a local first; only commit once the Linear forward and
        # the tanh succeeded.
        new_hidden = math.tanh(self.linear.forward(values + [self.hidden]))
        self.hidden = new_hidden
        self.outputs.append(new_hidden)
        # Continuing the trajectory stepwise mixes it with any recorded batch
        # pass, so that cache can no longer be back-propagated safely.
        self._fwd = None
        return new_hidden

    def _stream_step(self, stream, values, segment_start):
        # One step inside an active stream session. The row and the boundary
        # flag are already validated; the step follows the same segment-
        # boundary rules as the batch traversal so the recorded trajectory
        # back-propagates with the shared formulas once finish_stream commits
        # it.
        i = len(stream["inputs"])
        truncate = stream["truncate"]
        truncate_boundary = truncate is not None and i % truncate == 0
        if truncate_boundary or segment_start:
            # Segment start. The first segment starts from the session's
            # initial hidden state; without carry every later segment resets
            # to zero; with carry it starts from a numeric copy of the
            # previous segment's last hidden value. That copy is detached: it
            # enters this step's local derivatives but no gradient crosses
            # back over the boundary. Explicit segment_start marks merge with
            # the truncate-produced boundaries.
            if i == 0:
                hidden = stream["initial_hidden"]
            elif stream["carry_hidden"]:
                hidden = float(stream["outputs"][-1])
            else:
                hidden = 0.0
            stream["boundaries"].add(i)
        else:
            hidden = self.hidden
        new_hidden = math.tanh(self.linear.forward(values + [hidden]))
        self.hidden = new_hidden
        self.outputs.append(new_hidden)
        stream["inputs"].append(values)
        stream["prev_hiddens"].append(hidden)
        stream["outputs"].append(new_hidden)
        return new_hidden

    def start_stream(self, initial_hidden=None, truncate=None, carry_hidden=False):
        # Same validation rules as the batch forward, applied before any
        # state, cache, or Linear record is touched, so a rejected argument
        # changes nothing.
        if truncate is not None:
            if isinstance(truncate, bool) or not isinstance(truncate, int) or truncate <= 0:
                raise ValueError("truncate must be a positive integer or None")
        if not isinstance(carry_hidden, bool):
            raise ValueError("carry_hidden must be a boolean")
        if initial_hidden is None:
            initial_hidden = 0.0
        else:
            initial_hidden = _read_number(initial_hidden, "initial_hidden")
        if self._stream is not None:
            raise RuntimeError(
                "a stream session is already active; call finish_stream first")
        # Opening a session invalidates any previously recorded backward
        # cache and starts a fresh output trajectory from the given initial
        # hidden state. The Linear's parameters, accumulated gradients, and
        # last-forward record are left untouched.
        self.hidden = initial_hidden
        self.outputs = []
        self._fwd = None
        self._stream = {
            "inputs": [],
            "prev_hiddens": [],
            "outputs": [],
            "boundaries": set(),
            "truncate": truncate,
            "carry_hidden": carry_hidden,
            "initial_hidden": initial_hidden,
            # Parameter state as of the session's forward traversal; a later
            # apply_gradients() must not affect backward() of this session.
            "weights": list(self.linear.weight),
        }
        return None

    def finish_stream(self):
        stream = self._stream
        if stream is None:
            raise RuntimeError(
                "finish_stream requires an active session; call start_stream first")
        # Commit the recorded trajectory as the cached forward pass, in the
        # same shape the batch traversal produces, so backward() and
        # backward_with_initial_hidden() reuse the existing formulas.
        self._fwd = {
            "inputs": stream["inputs"],
            "prev_hiddens": stream["prev_hiddens"],
            "outputs": stream["outputs"],
            "boundaries": stream["boundaries"],
            "truncate": stream["truncate"],
            "carry_hidden": stream["carry_hidden"],
            "weights": stream["weights"],
        }
        self._stream = None
        return list(stream["outputs"])

    def forward(self, rows, truncate=None, carry_hidden=False, initial_hidden=None,
                segment_starts=None):
        if truncate is not None:
            if isinstance(truncate, bool) or not isinstance(truncate, int) or truncate <= 0:
                raise ValueError("truncate must be a positive integer or None")
        # Only an actual boolean is accepted (bool is a subclass of int, so an
        # explicit isinstance check is required). Validated before any state,
        # cache, or Linear forward record is touched.
        if not isinstance(carry_hidden, bool):
            raise ValueError("carry_hidden must be a boolean")
        # Optional external starting hidden state for this call. Omitted
        # (None) keeps the existing rule of starting from 0.0. Validated with
        # the same scalar rules as every other numeric argument, before any
        # state, cache, or Linear forward record is touched, so a rejected
        # value changes nothing.
        if initial_hidden is None:
            initial_hidden = 0.0
        else:
            initial_hidden = _read_number(initial_hidden, "initial_hidden")
        # Optional per-row boundary declarations, read in input-row order.
        # None keeps the existing behavior; otherwise one bool per row is
        # required, validated (type and length) before any state is touched.
        if segment_starts is not None:
            segment_flags = _read_bool_sequence(segment_starts, "segment_starts")
            try:
                n_rows = len(rows)
            except TypeError:
                raise ValueError(
                    "segment_starts requires a sized rows sequence")
            if len(segment_flags) != n_rows:
                raise ValueError(
                    "segment_starts length %d does not match rows length %d"
                    % (len(segment_flags), n_rows))
        else:
            segment_flags = None

        # Validate every input row (type, width and finite domain) up front,
        # before any state, cache or Linear forward record is touched, so a
        # batch containing a bad row anywhere can never leave a partial
        # trajectory.
        validated_rows = [_read_row(row, self.d) for row in rows]

        # Snapshot every piece of observable state the traversal may touch so
        # that any failure rolls the sequence back to its pre-call state: a
        # rejected traversal must leave neither a half-advanced hidden state
        # nor a partial Linear forward record, and must leave any earlier
        # successful forward cache available for backward().
        saved_hidden = self.hidden
        saved_outputs = self.outputs
        saved_fwd = self._fwd
        saved_stream = self._stream
        saved_last = self.linear.last
        saved_last_weight = self.linear._last_weight
        saved_grad = list(self.linear.grad)
        saved_grad_bias = self.linear.grad_bias

        inputs = []
        prev_hiddens = []
        outputs = []
        boundaries = set()
        # Parameter state of this forward pass. A later apply_gradients()
        # must not affect backward() of the recorded pass.
        weights = list(self.linear.weight)

        # The traversal works on locals only; self.* is committed solely on
        # full success below. An empty rows sequence therefore commits
        # initial_hidden as the current hidden state with empty outputs.
        hidden = initial_hidden
        try:
            for i, x in enumerate(validated_rows):
                declared = segment_flags is not None and segment_flags[i]
                if (truncate is not None and i % truncate == 0) or declared:
                    # Segment start. The very first segment starts from
                    # initial_hidden; without carry every later segment is
                    # reset to zero; with carry every later segment starts
                    # from a numeric copy of the previous segment's last
                    # hidden value. The value is detached from the autograd
                    # graph: it enters the local derivatives of this
                    # segment's first step but no gradient crosses back.
                    # Explicit segment_starts declarations merge with the
                    # truncate-produced boundaries; a mark at position 0 only
                    # confirms the initial boundary.
                    if i > 0 and carry_hidden:
                        hidden = outputs[-1]
                    elif i > 0:
                        hidden = 0.0
                    else:
                        hidden = initial_hidden
                    boundaries.add(i)
                prev_hiddens.append(hidden)
                # Linear.forward rejects a non-finite pre-activation before
                # caching anything, and math.tanh of a finite value is always
                # finite, so every committed hidden/output value is finite.
                hidden = math.tanh(self.linear.forward(x + [hidden]))
                inputs.append(x)
                outputs.append(hidden)
        except (TypeError, ValueError):
            self.hidden = saved_hidden
            self.outputs = saved_outputs
            self._fwd = saved_fwd
            self._stream = saved_stream
            self.linear.last = saved_last
            self.linear._last_weight = saved_last_weight
            self.linear.grad = saved_grad
            self.linear.grad_bias = saved_grad_bias
            raise

        self.hidden = hidden
        # Each consumer gets its own list: the public outputs field, the
        # backward cache and the value returned to the caller are independent
        # copies, so assigning, appending, deleting or clearing elements of the
        # returned list (or of seq.outputs) can never reach the cached
        # trajectory used by backward(), checkpoint() or export_state(). The
        # elements themselves are immutable numbers, so a shallow copy fully
        # detaches the lists. Two successive calls likewise never share a
        # returned object.
        self.outputs = list(outputs)
        # A successful batch traversal supersedes any half-finished stream
        # session; a failed one restored it above.
        self._stream = None
        # Cache of this forward pass for backward(). Only committed once the
        # whole traversal succeeded, so an illegal row leaves any earlier
        # successful record intact.
        self._fwd = {
            "inputs": inputs,
            "prev_hiddens": prev_hiddens,
            "outputs": list(outputs),
            "boundaries": boundaries,
            "truncate": truncate,
            "carry_hidden": carry_hidden,
            "weights": weights,
        }
        return list(outputs)

    def backward(self, grad_outputs):
        # Terminal hidden-state gradient is zero, so this is exactly the
        # previously recorded behavior: only the per-input gradient list is
        # returned (parameter gradients are accumulated on the Linear).
        return self._backward(grad_outputs, 0.0, return_initial=False)

    def backward_with_initial_hidden(self, grad_outputs, grad_hidden=0.0):
        # Like backward(), but additionally treats grad_hidden as the upstream
        # gradient of the cached forward's final hidden state and returns the
        # gradient with respect to that forward's initial_hidden as well.
        return self._backward(grad_outputs, grad_hidden, return_initial=True)

    def backward_with_boundaries(self, grad_outputs, grad_hidden=0.0):
        # Like backward_with_initial_hidden(), but additionally reports the
        # hidden-state gradient arriving at every segment start past index 0.
        # The third return value is a list of (index, gradient) pairs in
        # ascending index order, built from the cached pass's merged boundary
        # set (truncate-produced and explicitly declared starts, deduplicated).
        # Each gradient is taken with respect to the detached boundary
        # constant consumed at that start: the numeric copy of the previous
        # segment's final hidden value when carry_hidden was true, the zero
        # constant otherwise. Index 0 is not listed; its gradient is the
        # returned grad_initial_hidden. With no nonzero boundaries (or an
        # empty cached sequence) the list is empty.
        return self._backward(grad_outputs, grad_hidden, return_initial=True,
                              return_boundaries=True)

    def _backward(self, grad_outputs, grad_hidden, return_initial,
                  return_boundaries=False):
        if self._stream is not None:
            raise RuntimeError(
                "backward requires the stream session to be finished first")
        if self._fwd is None:
            raise RuntimeError("backward requires a cached forward pass; call forward first")

        inputs = self._fwd["inputs"]
        prev_hiddens = self._fwd["prev_hiddens"]
        outputs = self._fwd["outputs"]
        boundaries = self._fwd["boundaries"]
        # Forward-time parameter snapshot: the first d weights pair with the
        # d input features of each recorded row, the last with the previous
        # hidden state.
        weights = self._fwd["weights"]
        d = len(weights) - 1
        w_inputs = weights[:d]
        w_hidden = weights[d]
        n = len(outputs)
        # Validate every argument before any gradient is accumulated, so a
        # rejected call leaves both the cache and the Linear's accumulated
        # gradients exactly as they were.
        grad_outputs = _read_grad_list(grad_outputs, n)
        grad_hidden = _read_number(grad_hidden, "grad_hidden")

        # Accumulate this pass entirely into locals and commit to the Linear
        # only once every produced value is finite, so an overflowing pass
        # never leaves partially accumulated parameter gradients.
        param_grad = list(self.linear.grad)
        param_grad_bias = self.linear.grad_bias

        input_grads = [0.0] * n
        # Hidden-state gradients recorded at segment starts past index 0, in
        # traversal (descending) order; reversed into ascending order below.
        boundary_grads = []

        # The terminal hidden-state gradient seeds the recurrence at the last
        # step, where it is added to that step's output gradient. A zero seed
        # reproduces backward() bit for bit.
        hidden_grad = float(grad_hidden)
        # An empty cached sequence has no steps: return the terminal gradient
        # unchanged (as provided) without touching any parameter gradient.
        grad_initial_hidden = grad_hidden if n == 0 else 0.0
        try:
            for t in range(n - 1, -1, -1):
                # Gradient from later steps *within this segment* (including the
                # terminal hidden-state gradient at the final step) still flows
                # into the current step together with its own output gradient.
                dh = grad_outputs[t] + hidden_grad
                d_pre = dh * (1.0 - outputs[t] * outputs[t])  # tanh derivative

                # Parameter gradients at this step are always accumulated,
                # including at segment starts (the detached carry value is
                # treated as a constant and enters this local derivative only).
                row = inputs[t]
                for k in range(d):
                    param_grad[k] += d_pre * row[k]
                param_grad[d] += d_pre * prev_hiddens[t]
                param_grad_bias += d_pre

                # Input gradients mirror the input shape: one row of d gradients
                # per step for a multi-feature sequence, and the original flat
                # scalar list when d == 1.
                row_grads = [d_pre * w for w in w_inputs]
                input_grads[t] = row_grads[0] if d == 1 else row_grads
                # Step 0 always consumes initial_hidden as its previous hidden
                # state. With truncation the boundary cut below severs later
                # segments, so this local term is the full gradient with respect
                # to initial_hidden; without truncation it is reached uncut.
                if t == 0:
                    grad_initial_hidden = d_pre * w_hidden
                if t in boundaries:
                    # Truncated BPTT: segment-start hidden values (zeroed or
                    # carried as detached numbers) are constants, so no gradient
                    # crosses back into the prior segment. For a start past
                    # index 0, d_pre * w_hidden is the total gradient the merged
                    # within-segment recurrence delivers to that detached
                    # boundary constant; record it before severing the link.
                    if return_boundaries and t > 0:
                        boundary_grads.append((t, d_pre * w_hidden))
                    hidden_grad = 0.0
                else:
                    hidden_grad = d_pre * w_hidden
        except OverflowError:
            raise ValueError("backward result must be finite")

        # Verify the whole pass produced finite numbers before committing any
        # of it: non-finite parameter gradients, input gradients, seeds or
        # boundary gradients are all rejected with state untouched.
        def _check(value):
            _require_finite(value, "gradient")

        for value in param_grad:
            _check(value)
        _check(param_grad_bias)
        _check(grad_initial_hidden)
        for value in input_grads:
            if isinstance(value, list):
                for item in value:
                    _check(item)
            else:
                _check(value)
        for _, value in boundary_grads:
            _check(value)

        self.linear.grad = param_grad
        self.linear.grad_bias = param_grad_bias

        if return_boundaries:
            boundary_grads.reverse()
            return input_grads, grad_initial_hidden, boundary_grads
        if return_initial:
            return input_grads, grad_initial_hidden
        return input_grads
