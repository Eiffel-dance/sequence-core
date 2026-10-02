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


def _read_row(row):
    """Validate a single-element numeric sequence and return its value."""
    if isinstance(row, (str, bytes, bytearray, memoryview)):
        raise ValueError("each input row must be a single-element numeric sequence")
    try:
        length = len(row)
    except TypeError:
        raise ValueError("each input row must be a single-element numeric sequence")
    if length != 1:
        raise ValueError("each input row must contain exactly one numeric value")
    value = row[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("the input value must be a number")
    return value


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
        self.linear = linear
        self.hidden = 0.0
        self.outputs = []
        self._fwd = None

    def reset(self):
        self.hidden = 0.0
        self._fwd = None

    def step(self, row):
        x = _read_row(row)
        self.hidden = math.tanh(self.linear.forward([x, self.hidden]))
        # Stepping invalidates any cached forward pass.
        self._fwd = None
        return self.hidden

    def forward(self, rows, truncate=None, carry_hidden=False, initial_hidden=None):
        if truncate is not None:
            if isinstance(truncate, bool) or not isinstance(truncate, int) or truncate <= 0:
                raise ValueError("truncate must be a positive integer or None")
        # Only an actual boolean is accepted (bool is a subclass of int, so an
        # explicit isinstance check is required). Validated before any state,
        # cache, or Linear forward record is touched.
        if not isinstance(carry_hidden, bool):
            raise ValueError("carry_hidden must be a boolean")
        # The optional starting hidden state uses the same scalar validation as
        # the other numeric arguments; None means "start from zero" as before.
        # Validated before any observable state is touched so a rejected call
        # leaves hidden, outputs, the forward cache and Linear's last record
        # exactly as they were.
        if initial_hidden is None:
            init_hidden = 0.0
        else:
            init_hidden = _read_number(initial_hidden, "initial_hidden")

        # Snapshot every piece of observable state the traversal may touch so
        # that any failure rolls the sequence back to its pre-call state: a
        # rejected traversal must leave neither a half-advanced hidden state
        # nor a partial Linear forward record, and must leave any earlier
        # successful forward cache available for backward().
        saved_hidden = self.hidden
        saved_outputs = self.outputs
        saved_fwd = self._fwd
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
        # full success below. The first step of the first segment starts from
        # the externally supplied hidden state (0.0 when omitted); this also
        # fixes the final hidden for an empty rows list.
        hidden = init_hidden
        try:
            for i, row in enumerate(rows):
                if truncate is not None and i % truncate == 0:
                    # Segment start. The first segment starts from
                    # initial_hidden; without carry every later segment is
                    # reset to zero; with carry each later segment starts from
                    # a numeric copy of the previous segment's last hidden
                    # value. The carried value is detached from the autograd
                    # graph: it enters the local derivatives of this segment's
                    # first step but no gradient crosses the boundary back.
                    if i == 0:
                        hidden = init_hidden
                    elif carry_hidden:
                        hidden = outputs[-1]
                    else:
                        hidden = 0.0
                    boundaries.add(i)
                x = _read_row(row)
                prev_hiddens.append(hidden)
                hidden = math.tanh(self.linear.forward([x, hidden]))
                inputs.append(x)
                outputs.append(hidden)
        except (TypeError, ValueError):
            self.hidden = saved_hidden
            self.outputs = saved_outputs
            self._fwd = saved_fwd
            self.linear.last = saved_last
            self.linear._last_weight = saved_last_weight
            self.linear.grad = saved_grad
            self.linear.grad_bias = saved_grad_bias
            raise

        self.hidden = hidden
        self.outputs = outputs
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
            "initial_hidden": init_hidden,
            "weights": weights,
        }
        return self.outputs

    def backward(self, grad_outputs):
        if self._fwd is None:
            raise RuntimeError("backward requires a cached forward pass; call forward first")

        inputs = self._fwd["inputs"]
        prev_hiddens = self._fwd["prev_hiddens"]
        outputs = self._fwd["outputs"]
        boundaries = self._fwd["boundaries"]
        w_input = self._fwd["weights"][0]
        w_hidden = self._fwd["weights"][1]
        n = len(outputs)
        grad_outputs = _read_grad_list(grad_outputs, n)

        input_grads = [0.0] * n

        # Hidden-state gradient propagated from step t + 1 back into step t.
        hidden_grad = 0.0
        for t in range(n - 1, -1, -1):
            # Gradient from later steps *within this segment* still flows
            # into the segment-start step itself.
            dh = grad_outputs[t] + hidden_grad
            d_pre = dh * (1.0 - outputs[t] * outputs[t])  # tanh derivative

            # Parameter gradients at this step are always accumulated,
            # including at segment starts (the hidden-state term is zero
            # there because the start hidden is reset to zero).
            self.linear.grad[0] += d_pre * inputs[t]
            self.linear.grad[1] += d_pre * prev_hiddens[t]
            self.linear.grad_bias += d_pre

            input_grads[t] = d_pre * w_input
            if t in boundaries:
                # Truncated BPTT: this step consumed a hidden state forced to
                # zero, so no gradient crosses back into the prior segment.
                hidden_grad = 0.0
            else:
                hidden_grad = d_pre * w_hidden

        return input_grads
