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

    def forward(self, x):
        values = _read_number_sequence(x, len(self.weight), "x")
        result = sum(w * a for w, a in zip(self.weight, values)) + self.bias
        # Cache only after validation and computation succeed. The cache
        # records the parameter state that produced this output so a later
        # backward() stays consistent even if apply_gradients() ran in
        # between; apply_gradients()/zero_grad() never invalidate it.
        self.last = {
            "x": values,
            "weight": list(self.weight),
            "bias": self.bias,
        }
        return result

    def backward(self, grad):
        if self.last is None:
            raise RuntimeError("backward requires a successful forward call first")
        _read_number(grad, "grad")
        # Gradients are accumulated item by item across repeated calls.
        self.grad = [g + grad * a for g, a in zip(self.grad, self.last["x"])]
        self.grad_bias += grad
        # Input gradients use the weights as they were during the cached
        # forward pass, not the possibly updated current weights.
        return [grad * w for w in self.last["weight"]]

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
        # Cache of this forward pass for backward(). The linear weights are
        # snapshotted so backward() reflects this pass even if the parameters
        # are updated before backward() runs.
        self._fwd = {
            "inputs": inputs,
            "prev_hiddens": prev_hiddens,
            "outputs": outputs,
            "boundaries": boundaries,
            "truncate": truncate,
            "w_input": self.linear.weight[0],
            "w_hidden": self.linear.weight[1],
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

        w_input = self._fwd["w_input"]
        w_hidden = self._fwd["w_hidden"]
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
