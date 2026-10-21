"""Train a 2-layer MLP (784-128-10) on MNIST using nothing but NumPy.

Usage:
    python train.py                       # default: Adam, lr=1e-3, 10 epochs
    python train.py --epochs 5 --optim sgd --lr 0.1
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from data import iterate_minibatches, load_mnist
from nn import SGD, Adam, Dropout, Linear, ReLU, Sequential, SoftmaxCrossEntropy


def build_model(hidden: int, dropout: float, seed: int) -> Sequential:
    layers = [Linear(784, hidden, init="he", seed=seed), ReLU()]
    if dropout > 0:
        layers.append(Dropout(dropout, seed=seed))
    layers.append(Linear(hidden, 10, init="he", seed=seed + 1))
    return Sequential(*layers)


def accuracy(model: Sequential, x: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(x) == y).mean())


def plot_curves(history: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], marker="o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Training loss")
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, np.array(history["train_acc"]) * 100, marker="o", label="train acc")
    ax2.plot(epochs, np.array(history["test_acc"]) * 100, marker="s", label="test acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved training curves to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=None, help="default: adam=1e-3, sgd=0.05")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="training_curve.png")
    parser.add_argument("--save", default="checkpoints/mlp_mnist.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"train {x_train.shape}, test {x_test.shape}")

    model = build_model(args.hidden, args.dropout, args.seed)
    print(model)
    print(f"parameters: {model.num_parameters():,}")

    if args.optim == "adam":
        opt = Adam(model.parameters(), lr=args.lr or 1e-3, weight_decay=args.weight_decay)
    else:
        opt = SGD(model.parameters(), lr=args.lr or 0.05, momentum=0.9, weight_decay=args.weight_decay)
    criterion = SoftmaxCrossEntropy()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, seen = 0.0, 0, 0
        for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, rng):
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            # Backprop: dL/dlogits first, then back through every layer
            model.backward(criterion.backward())
            opt.step()
            opt.zero_grad()

            total_loss += loss * len(xb)
            total_correct += int((logits.argmax(axis=1) == yb).sum())
            seen += len(xb)

        train_loss = total_loss / seen
        train_acc = total_correct / seen
        test_acc = accuracy(model, x_test, y_test)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | "
            f"train acc {train_acc * 100:.2f}% | test acc {test_acc * 100:.2f}% | {time.time() - t0:.1f}s"
        )

    best = max(history["test_acc"]) * 100
    print(f"Final test accuracy {history['test_acc'][-1] * 100:.2f}% (best {best:.2f}%)")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        np.savez(args.save, **model.state_dict())
        print(f"Saved weights to {args.save}")
    if args.plot:
        plot_curves(history, args.plot)


if __name__ == "__main__":
    main()
