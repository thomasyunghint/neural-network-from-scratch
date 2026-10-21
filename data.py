"""MNIST loader: download the raw idx files -> parse into numpy -> cache as .npz.

No torchvision dependency; only the standard library and numpy.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

# Same mirrors torchvision uses; more reliable than yann.lecun.com
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

# Mean and std of MNIST pixels scaled to [0, 1], used for standardization
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _download(filename: str, root: str) -> str:
    path = os.path.join(root, filename)
    if os.path.exists(path):
        return path
    os.makedirs(root, exist_ok=True)
    last_err: Exception | None = None
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"could not download {filename}") from last_err


def _parse_idx(path: str) -> np.ndarray:
    """Parse the idx format: magic (4B) + one 4B length per dimension + raw data."""
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(root: str = "data", normalize: bool = True, flatten: bool = True):
    """Return (x_train, y_train, x_test, y_test).

    x: float32, standardized and flattened to (N, 784) by default
    y: int64 class indices of shape (N,)
    """
    cache = os.path.join(root, "mnist.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        arrays = {k: d[k] for k in FILES}
    else:
        arrays = {k: _parse_idx(_download(fn, root)) for k, fn in FILES.items()}
        os.makedirs(root, exist_ok=True)
        np.savez_compressed(cache, **arrays)

    x_train = arrays["x_train"].astype(np.float32) / 255.0
    x_test = arrays["x_test"].astype(np.float32) / 255.0
    if normalize:
        x_train = (x_train - MNIST_MEAN) / MNIST_STD
        x_test = (x_test - MNIST_MEAN) / MNIST_STD
    if flatten:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)
    return x_train, arrays["y_train"].astype(np.int64), x_test, arrays["y_test"].astype(np.int64)


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def iterate_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    """Yield (xb, yb) minibatches, optionally reshuffled every epoch."""
    idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for start in range(0, len(x), batch_size):
        batch = idx[start:start + batch_size]
        yield x[batch], y[batch]


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_mnist()
    print("train:", xtr.shape, ytr.shape, "test:", xte.shape, yte.shape)
    print("label counts:", np.bincount(ytr))
