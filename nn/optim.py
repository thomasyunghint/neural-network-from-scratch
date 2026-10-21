"""Optimizers: SGD (with momentum) and Adam.

Each optimizer holds a list of Parameters and updates param.data in place
when step() is called.
"""

from __future__ import annotations

import numpy as np

from .layers import Parameter


class Optimizer:
    def __init__(self, params: list[Parameter], lr: float, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def _grad(self, p: Parameter) -> np.ndarray:
        """Gradient including the L2 (weight decay) term."""
        if self.weight_decay:
            return p.grad + self.weight_decay * p.data
        return p.grad


class SGD(Optimizer):
    """Stochastic gradient descent with momentum:
        v <- momentum * v + g
        w <- w - lr * v
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            g = self._grad(p)
            if self.momentum:
                v *= self.momentum
                v += g
                p.data -= self.lr * v
            else:
                p.data -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015):
        m <- b1 * m + (1 - b1) * g
        v <- b2 * v + (1 - b2) * g^2
        m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)   (bias correction)
        w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = self._grad(p)
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
            m_hat = m / bias1
            v_hat = v / bias2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
