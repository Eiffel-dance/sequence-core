import math
from collections.abc import Sequence


def _read_number(value, what):
    """Validate a scalar argument; bool is not accepted as a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a Python int or float" % what)
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
        values.append(value)
    return values


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
        result = sum(w * a for w, a in zip(self.weight, values)) + self.bias
        # Cache only after validation and computation succeed. Record the
        # inputs together with the exact weights that produced the output.
        self.last = values
        self._last_weight = list(self.weight)
        return result

    def backward(self, grad):
        if self.last is None:
            raise RuntimeError("backward requires a successful forward call first")
        _read_number(grad, "grad")
        # Gradients are accumulated item by item across repeated calls, using
        # the cached inputs of the recorded forward pass.
        self.grad = [g + grad * a for g, a in zip(self.grad, self.last)]
        self.grad_bias += grad
        # Input gradients must use the weights as they were at forward time.
        return [grad * w for w in self._last_weight]

    def zero_grad(self):
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0

    def apply_gradients(self, learning_rate):
        _read_number(learning_rate, "learning_rate")
        new_weight = [w - learning_rate * g for w, g in zip(self.weight, self.grad)]
        new_bias = self.bias - learning_rate * self.grad_bias
        self.weight = new_weight
        self.bias = new_bias


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

    def step(self, row):
        # Validate before any state is touched, so a rejected row changes
        # neither hidden/outputs nor the Linear's last-forward record, and a
        # batch forward cache stays available for backward().
        values = _read_row(row, self.d)
        stream = self._stream
        if stream is not None:
            return self._stream_step(stream, values)
        # Compute into a local first; only commit once the Linear forward and
        # the tanh succeeded.
        new_hidden = math.tanh(self.linear.forward(values + [self.hidden]))
        self.hidden = new_hidden
        self.outputs.append(new_hidden)
        # Continuing the trajectory stepwise mixes it with any recorded batch
        # pass, so that cache can no longer be back-propagated safely.
        self._fwd = None
        return new_hidden

    def _stream_step(self, stream, values):
        # One step inside an active stream session. The row is already
        # validated; the step follows the same segment-boundary rules as the
        # batch traversal so the recorded trajectory back-propagates with the
        # shared formulas once finish_stream commits it.
        i = len(stream["inputs"])
        truncate = stream["truncate"]
        if truncate is not None and i % truncate == 0:
            # Segment start. The first segment starts from the session's
            # initial hidden state; without carry every later segment resets
            # to zero; with carry it starts from a numeric copy of the
            # previous segment's last hidden value. That copy is detached: it
            # enters this step's local derivatives but no gradient crosses
            # back over the boundary.
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

    def forward(self, rows, truncate=None, carry_hidden=False, initial_hidden=None):
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
            for i, row in enumerate(rows):
                if truncate is not None and i % truncate == 0:
                    # Segment start. The very first segment starts from
                    # initial_hidden; without carry every later segment is
                    # reset to zero; with carry every later segment starts
                    # from a numeric copy of the previous segment's last
                    # hidden value. The value is detached from the autograd
                    # graph: it enters the local derivatives of this
                    # segment's first step but no gradient crosses back.
                    if i > 0 and carry_hidden:
                        hidden = outputs[-1]
                    elif i > 0:
                        hidden = 0.0
                    else:
                        hidden = initial_hidden
                    boundaries.add(i)
                x = _read_row(row, self.d)
                prev_hiddens.append(hidden)
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
        self.outputs = outputs
        # A successful batch traversal supersedes any half-finished stream
        # session; a failed one restored it above.
        self._stream = None
        # Cache of this forward pass for backward(). Only committed once the
        # whole traversal succeeded, so an illegal row leaves any earlier
        # successful record intact.
        self._fwd = {
            "inputs": inputs,
            "prev_hiddens": prev_hiddens,
            "outputs": outputs,
            "boundaries": boundaries,
            "truncate": truncate,
            "carry_hidden": carry_hidden,
            "weights": weights,
        }
        return self.outputs

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

    def _backward(self, grad_outputs, grad_hidden, return_initial):
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

        input_grads = [0.0] * n

        # The terminal hidden-state gradient seeds the recurrence at the last
        # step, where it is added to that step's output gradient. A zero seed
        # reproduces backward() bit for bit.
        hidden_grad = float(grad_hidden)
        # An empty cached sequence has no steps: return the terminal gradient
        # unchanged (as provided) without touching any parameter gradient.
        grad_initial_hidden = grad_hidden if n == 0 else 0.0
        for t in range(n - 1, -1, -1):
            # Gradient from later steps *within this segment* (including the
            # terminal hidden-state gradient at the final step) still flows
            # into the current step together with its own output gradient.
            dh = grad_outputs[t] + hidden_grad
            d_pre = dh * (1.0 - outputs[t] * outputs[t])  # tanh derivative

            # Parameter gradients at this step are always accumulated,
            # including at segment starts (the detached carry value is treated
            # as a constant and enters this local derivative only).
            row = inputs[t]
            for k in range(d):
                self.linear.grad[k] += d_pre * row[k]
            self.linear.grad[d] += d_pre * prev_hiddens[t]
            self.linear.grad_bias += d_pre

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
                # crosses back into the prior segment.
                hidden_grad = 0.0
            else:
                hidden_grad = d_pre * w_hidden

        if return_initial:
            return input_grads, grad_initial_hidden
        return input_grads
