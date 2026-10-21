"""Numerical gradient checks: compare analytic gradients from backward()
against central finite differences.

For any scalar function f(x),
    df/dx_i ~= (f(x + h e_i) - f(x - h e_i)) / (2h)
With float64 and h = 1e-5 the error should be well below 1e-6.

To turn a layer with vector output into a scalar function we dot the output
with a fixed random matrix R:
    f(x) = sum(layer(x) * R)   =>   df/dy = R
so layer.backward(R) must equal the numerical gradient of f with respect to
the input (or the parameters).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import Dropout, Linear, ReLU, Sequential, Sigmoid, SoftmaxCrossEntropy, Tanh  # noqa: E402

H = 1e-5
TOL = 1e-6


def numerical_grad(f, x: np.ndarray, h: float = H) -> np.ndarray:
    """Central differences over every element of x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        fp = f()
        x[idx] = old - h
        fm = f()
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * h)
    return grad


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def _check_layer_input_grad(layer, x: np.ndarray, rng: np.random.Generator) -> None:
    """Check dL/dx."""
    out = layer.forward(x)
    R = rng.standard_normal(out.shape)
    analytic = layer.backward(R)
    numeric = numerical_grad(lambda: np.sum(layer.forward(x) * R), x)
    assert rel_error(analytic, numeric) < TOL


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.mark.parametrize("act_cls", [ReLU, Sigmoid, Tanh])
def test_activation_grad(act_cls, rng):
    layer = act_cls()
    # Keep inputs away from 0 where ReLU is not differentiable
    x = rng.standard_normal((4, 7))
    x[np.abs(x) < 1e-2] += 0.1
    _check_layer_input_grad(layer, x, rng)


def test_linear_grads(rng):
    layer = Linear(5, 3, seed=0)
    # Use float64 so the numerical gradient is precise enough
    layer.W.data = layer.W.data.astype(np.float64)
    layer.b.data = layer.b.data.astype(np.float64)
    x = rng.standard_normal((4, 5))
    out = layer.forward(x)
    R = rng.standard_normal(out.shape)

    dx = layer.backward(R)
    dW, db = layer.W.grad.copy(), layer.b.grad.copy()

    f = lambda: np.sum(layer.forward(x) * R)  # noqa: E731
    assert rel_error(dx, numerical_grad(f, x)) < TOL
    assert rel_error(dW, numerical_grad(f, layer.W.data)) < TOL
    assert rel_error(db, numerical_grad(f, layer.b.data)) < TOL


def test_dropout_grad_train_and_eval(rng):
    layer = Dropout(p=0.3, seed=1)
    x = rng.standard_normal((6, 5))

    # Training mode: the mask is sampled in forward, so compare against the cached mask
    layer.train()
    layer.forward(x)
    mask = layer.cache.copy()
    R = rng.standard_normal(x.shape)
    analytic = layer.backward(R)
    assert np.allclose(analytic, R * mask)

    # Eval mode: identity
    layer.eval()
    _check_layer_input_grad(layer, x, rng)


def test_softmax_cross_entropy_grad(rng):
    criterion = SoftmaxCrossEntropy()
    logits = rng.standard_normal((5, 4)) * 3.0
    targets = rng.integers(0, 4, size=5)

    criterion.forward(logits, targets)
    analytic = criterion.backward()
    numeric = numerical_grad(lambda: criterion.forward(logits, targets), logits)
    assert rel_error(analytic, numeric) < TOL


def test_softmax_cross_entropy_numerically_stable():
    """Huge logits must not produce inf / nan."""
    criterion = SoftmaxCrossEntropy()
    logits = np.array([[1000.0, -1000.0, 0.0]])
    loss = criterion.forward(logits, np.array([0]))
    assert np.isfinite(loss) and loss < 1e-6
    assert np.all(np.isfinite(criterion.backward()))


def test_softmax_accepts_one_hot(rng):
    criterion = SoftmaxCrossEntropy()
    logits = rng.standard_normal((3, 4))
    targets = np.array([0, 2, 3])
    onehot = np.eye(4)[targets]
    assert np.isclose(criterion.forward(logits, targets), criterion.forward(logits, onehot))


def test_full_mlp_end_to_end_grad(rng):
    """Gradient-check the whole network plus loss: verifies Sequential chains the layers correctly."""
    model = Sequential(Linear(6, 8, seed=0), Tanh(), Linear(8, 5, seed=1), ReLU(), Linear(5, 3, seed=2))
    for p in model.parameters():
        p.data = p.data.astype(np.float64)
    criterion = SoftmaxCrossEntropy()

    x = rng.standard_normal((4, 6))
    y = rng.integers(0, 3, size=4)

    def loss_fn() -> float:
        return criterion.forward(model.forward(x), y)

    loss_fn()
    dx = model.backward(criterion.backward())
    param_grads = [p.grad.copy() for p in model.parameters()]

    assert rel_error(dx, numerical_grad(loss_fn, x)) < TOL
    for p, g in zip(model.parameters(), param_grads):
        assert rel_error(g, numerical_grad(loss_fn, p.data)) < TOL
