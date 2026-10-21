"""Optimizer sanity tests: each optimizer should minimize a simple quadratic."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import SGD, Adam, Parameter  # noqa: E402


@pytest.mark.parametrize("make_opt", [
    lambda p: SGD([p], lr=0.1),
    lambda p: SGD([p], lr=0.1, momentum=0.9),
    lambda p: Adam([p], lr=0.1),
])
def test_optimizer_minimizes_quadratic(make_opt):
    """Minimize f(w) = ||w - target||^2, whose gradient is 2(w - target)."""
    target = np.array([1.0, -2.0, 3.0])
    w = Parameter(np.zeros(3))
    opt = make_opt(w)
    for _ in range(300):
        w.grad = 2.0 * (w.data - target)
        opt.step()
        opt.zero_grad()
    assert np.allclose(w.data, target, atol=1e-2)


def test_zero_grad_clears_gradients():
    w = Parameter(np.ones(4))
    w.grad = np.full(4, 7.0)
    SGD([w], lr=0.1).zero_grad()
    assert np.all(w.grad == 0)


def test_weight_decay_shrinks_weights():
    w = Parameter(np.ones(3))
    opt = SGD([w], lr=0.1, weight_decay=0.5)
    opt.step()  # grad is 0, so only the L2 term acts: w <- w - lr * wd * w
    assert np.allclose(w.data, 0.95)
