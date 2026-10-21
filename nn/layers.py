"""Core neural network layers. Every layer implements forward / backward.

Conventions:
- Inputs x have shape (N, D) where N is the batch size.
- forward caches whatever backward needs in self.cache.
- backward receives the upstream gradient dL/dy (same shape as the forward
  output), returns dL/dx (same shape as the forward input), and writes
  parameter gradients into Parameter.grad.
"""

from __future__ import annotations

import numpy as np


class Parameter:
    """A trainable tensor holding both its value (data) and gradient (grad)."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Parameter(shape={self.data.shape})"


class Layer:
    """Base class. Parameter-free layers only need forward / backward."""

    def __init__(self):
        self.training = True
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        return []

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Linear(Layer):
    """Fully connected layer: y = x @ W + b

    Backward pass:
        dL/dW = x^T @ dL/dy
        dL/db = sum over the batch of dL/dy
        dL/dx = dL/dy @ W^T
    """

    def __init__(self, in_features: int, out_features: int, init: str = "he", seed: int | None = None):
        super().__init__()
        rng = np.random.default_rng(seed)
        if init == "he":  # for ReLU: Var(W) = 2 / fan_in
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":  # for sigmoid / tanh: Var(W) = 1 / fan_in
            std = np.sqrt(1.0 / in_features)
        else:
            raise ValueError(f"unknown init scheme: {init}")
        self.W = Parameter(rng.normal(0.0, std, size=(in_features, out_features)))
        self.b = Parameter(np.zeros(out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.W.data + self.b.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self.cache
        self.W.grad = x.T @ grad_out
        self.b.grad = grad_out.sum(axis=0)
        return grad_out @ self.W.data.T

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]


class ReLU(Layer):
    """ReLU(x) = max(0, x); derivative is 1[x > 0]."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x > 0
        return x * self.cache

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.cache


class Sigmoid(Layer):
    """sigmoid(x) = 1 / (1 + e^{-x}); derivative is s(1 - s)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Piecewise evaluation avoids overflow in exp
        out = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, -500, None))),
                       np.exp(np.clip(x, None, 500)) / (1.0 + np.exp(np.clip(x, None, 500))))
        self.cache = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        s = self.cache
        return grad_out * s * (1.0 - s)


class Tanh(Layer):
    """tanh(x); derivative is 1 - tanh^2(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.cache = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        t = self.cache
        return grad_out * (1.0 - t * t)


class Dropout(Layer):
    """Inverted dropout: zero units with probability p and scale by 1/(1-p)
    during training; identity at inference time."""

    def __init__(self, p: float = 0.5, seed: int | None = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self.cache = None
            return x
        mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        self.cache = mask
        return x * mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.cache is None:
            return grad_out
        return grad_out * self.cache


class SoftmaxCrossEntropy:
    """Fused softmax + cross-entropy loss (numerically stable).

    forward(logits, targets) returns the mean loss over the batch:
        L = -1/N * sum_i log softmax(z_i)[y_i]
    log softmax is computed with the log-sum-exp trick so exp never overflows.

    backward() returns dL/dlogits = (softmax(z) - onehot(y)) / N, which is the
    strikingly simple gradient you get when softmax and cross-entropy are fused.
    """

    def __init__(self):
        self.cache = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        # targets may be class indices (N,) or one-hot vectors (N, C)
        if targets.ndim == 2:
            targets = targets.argmax(axis=1)
        n = logits.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss = -log_probs[np.arange(n), targets].mean()
        self.cache = (np.exp(log_probs), targets)
        return float(loss)

    def backward(self) -> np.ndarray:
        probs, targets = self.cache
        n = probs.shape[0]
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0
        return grad / n

    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> float:
        return self.forward(logits, targets)
