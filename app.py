import math
from collections.abc import Sequence


_TEXT_BYTES = (str, bytes, bytearray, memoryview)


def _is_number(value):
    """Return True only for plain Python int or float (bool is rejected)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_numeric_sequence(values, what):
    """Validate a non-text, non-bytes sequence of int/float values.

    Returns a fresh list preserving the incoming order. Raises ValueError
    for text, byte-like values, non-sequences, or illegal elements.
    """
    if isinstance(values, _TEXT_BYTES) or not isinstance(values, Sequence):
        raise ValueError("%s must be a non-text sequence of numbers" % what)
    result = []
    for value in values:
        if not _is_number(value):
            raise ValueError("each %s value must be an int or float" % what)
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
        # Validate completely before any attribute is bound, so a rejected
        # construction never leaves a partially initialized layer behind.
        checked_weight = _read_numeric_sequence(weight, "weight")
        if not _is_number(bias):
            raise ValueError("bias must be an int or float")
        self.weight = checked_weight
        self.bias = bias
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0
        self._last = None

    def forward(self, x):
        # Fully validate first: a failed call must not touch the cache.
        values = _read_numeric_sequence(x, "input")
        if len(values) != len(self.weight):
            raise ValueError(
                "input length %d does not match weight length %d"
                % (len(values), len(self.weight))
            )
        # Keep a defensive copy of the accepted input for backward.
        self._last = list(values)
        return sum(w * a for w, a in zip(self.weight, values)) + self.bias

    def backward(self, grad):
        # Missing forward is a state error and takes priority over the type
        # of grad, matching the pre-existing RuntimeError contract.
        if self._last is None:
            raise RuntimeError("backward requires a cached forward pass; call forward first")
        if not _is_number(grad):
            raise ValueError("grad must be an int or float")
        # Gradients are accumulated item by item across repeated calls.
        last = self._last
        self.grad = [g + grad * a for g, a in zip(self.grad, last)]
        self.grad_bias += grad
        return [grad * w for w in self.weight]

    def zero_grad(self):
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0

    def apply_gradients(self, learning_rate):
        # Reject illegal learning rates before mutating any parameter.
        if not _is_number(learning_rate):
            raise ValueError("learning_rate must be an int or float")
        for j in range(len(self.weight)):
            self.weight[j] = self.weight[j] - learning_rate * self.grad[j]
        self.bias = self.bias - learning_rate * self.grad_bias


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

    def forward(self, rows, truncate=None):
        if truncate is not None:
            if isinstance(truncate, bool) or not isinstance(truncate, int) or truncate <= 0:
                raise ValueError("truncate must be a positive integer or None")

        inputs = []
        prev_hiddens = []
        outputs = []
        boundaries = set()

        self.hidden = 0.0
        for i, row in enumerate(rows):
            if truncate is not None and i % truncate == 0:
                # Segment start: hidden state is reset to zero.
                self.hidden = 0.0
                boundaries.add(i)
            x = _read_row(row)
            prev_hiddens.append(self.hidden)
            self.hidden = math.tanh(self.linear.forward([x, self.hidden]))
            inputs.append(x)
            outputs.append(self.hidden)

        self.outputs = outputs
        # Cache of this forward pass for backward().
        self._fwd = {
            "inputs": inputs,
            "prev_hiddens": prev_hiddens,
            "outputs": outputs,
            "boundaries": boundaries,
            "truncate": truncate,
        }
        return self.outputs

    def backward(self, grad_outputs):
        if self._fwd is None:
            raise RuntimeError("backward requires a cached forward pass; call forward first")

        inputs = self._fwd["inputs"]
        prev_hiddens = self._fwd["prev_hiddens"]
        outputs = self._fwd["outputs"]
        boundaries = self._fwd["boundaries"]
        n = len(outputs)
        grad_outputs = _read_grad_list(grad_outputs, n)

        w_input = self.linear.weight[0]
        w_hidden = self.linear.weight[1]
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
