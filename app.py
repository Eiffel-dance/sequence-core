import math


class Linear:
    def __init__(self, weight, bias=0.0):
        self.weight = list(weight)
        self.bias = bias
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0

    def forward(self, x):
        self.last = x
        return sum(w * a for w, a in zip(self.weight, x)) + self.bias

    def backward(self, grad):
        # Weight and bias gradients accumulate across repeated calls.
        for j, a in enumerate(self.last):
            self.grad[j] += grad * a
        self.grad_bias += grad
        return [grad * w for w in self.weight]

    def zero_grad(self):
        self.grad = [0.0] * len(self.weight)
        self.grad_bias = 0.0

    def apply_gradients(self, learning_rate):
        # Gradients are intentionally kept after the update so callers
        # decide when to clear them via zero_grad().
        for j in range(len(self.weight)):
            self.weight[j] = self.weight[j] - learning_rate * self.grad[j]
        self.bias = self.bias - learning_rate * self.grad_bias


class TanhSequence:
    def __init__(self, linear):
        self.linear = linear
        self.hidden = 0.0
        self.outputs = []
        self._cache = None

    @staticmethod
    def _single_value(row):
        try:
            values = list(row)
        except TypeError:
            raise ValueError("each input row must be a single-element numeric sequence")
        if len(values) != 1 or not isinstance(values[0], (int, float)) or isinstance(values[0], bool):
            raise ValueError("each input row must be a single-element numeric sequence")
        return float(values[0])

    def reset(self):
        self.hidden = 0.0
        self._cache = None

    def step(self, row):
        value = self._single_value(row)
        # Any incremental stepping invalidates a previous forward() cache.
        self._cache = None
        self.hidden = math.tanh(self.linear.forward([value, self.hidden]))
        return self.hidden

    def forward(self, rows, truncate=None):
        if truncate is not None and (
            not isinstance(truncate, int) or isinstance(truncate, bool) or truncate <= 0
        ):
            raise ValueError("truncate must be None or a positive integer")
        values = [self._single_value(row) for row in rows]

        self._cache = None
        self.hidden = 0.0
        inputs, hidden_in, outputs = [], [], []
        for i, value in enumerate(values):
            if truncate and i % truncate == 0:
                self.hidden = 0.0
            inputs.append(value)
            hidden_in.append(self.hidden)
            self.hidden = math.tanh(self.linear.forward([value, self.hidden]))
            outputs.append(self.hidden)

        self.outputs = outputs
        self._cache = {
            "inputs": inputs,
            "hidden_in": hidden_in,
            "outputs": outputs,
            "truncate": truncate,
        }
        return outputs

    def backward(self, output_grads):
        if self._cache is None:
            raise RuntimeError("backward requires a recent forward() cache")
        try:
            grads = list(output_grads)
        except TypeError:
            raise ValueError("output_grads must be a list matching the cached sequence")
        cache = self._cache
        n = len(cache["outputs"])
        if len(grads) != n:
            raise ValueError("output_grads length does not match the cached sequence length")

        truncate = cache["truncate"]
        outputs = cache["outputs"]
        inputs = cache["inputs"]
        hidden_in = cache["hidden_in"]
        input_grads = [0.0] * n
        dh_next = 0.0
        for i in range(n - 1, -1, -1):
            h = outputs[i]
            ds = (grads[i] + dh_next) * (1.0 - h * h)
            # Linear.forward only retains its most recent input, so restore
            # this step's cached inputs for correct per-step gradients.
            self.linear.last = [inputs[i], hidden_in[i]]
            gx = self.linear.backward(ds)
            input_grads[i] = gx[0]
            # At a segment start, no hidden-state gradient flows into the
            # previous segment; ds above still accumulated this step's own
            # parameter gradient.
            if truncate and i % truncate == 0:
                dh_next = 0.0
            else:
                dh_next = gx[1]
        return input_grads
