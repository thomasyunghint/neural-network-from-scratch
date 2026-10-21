"""Sequential container: chains layers, runs forward in order and backward in reverse."""

from __future__ import annotations

import numpy as np

from .layers import Layer, Parameter


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Chain rule: propagate the gradient from the last layer back to the first
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def parameters(self) -> list[Parameter]:
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self) -> None:
        self.training = True
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            layer.eval()

    def predict(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Batched inference; returns predicted class indices."""
        was_training = self.training
        self.eval()
        preds = [self.forward(x[i:i + batch_size]).argmax(axis=1) for i in range(0, len(x), batch_size)]
        if was_training:
            self.train()
        return np.concatenate(preds)

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def state_dict(self) -> dict[str, np.ndarray]:
        return {f"param_{i}": p.data for i, p in enumerate(self.parameters())}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for i, p in enumerate(self.parameters()):
            p.data[...] = state[f"param_{i}"]

    def __repr__(self) -> str:
        inner = "\n".join(f"  ({i}) {layer.__class__.__name__}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{inner}\n)"
