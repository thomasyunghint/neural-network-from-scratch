# Neural Network from Scratch (pure NumPy)

I wanted to understand what actually happens when you call `loss.backward()`, so I built a small deep learning framework with nothing but NumPy: layers with hand-written forward and backward passes, SGD and Adam optimizers, and a training loop. A two-layer MLP trained with it gets **97.4% test accuracy on MNIST**, and every `backward` is checked against numerical gradients so I know the math is right, not just that the loss goes down.

![training curves](training_curve.png)

## What's inside

```
nn/
├── layers.py   # Linear, ReLU, Sigmoid, Tanh, Dropout, SoftmaxCrossEntropy
├── optim.py    # SGD with momentum, Adam
└── model.py    # Sequential container
data.py         # downloads MNIST, parses the raw idx files, normalizes, batches
train.py        # trains the MLP, prints per-epoch metrics, saves the curves above
tests/
├── test_gradcheck.py  # finite-difference checks for every layer and the full network
└── test_optim.py      # optimizers actually minimize a simple function
```

The API looks like a very small PyTorch, on purpose. Building and training a model is a few lines:

```python
from nn import Sequential, Linear, ReLU, SoftmaxCrossEntropy, Adam

model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
criterion = SoftmaxCrossEntropy()
opt = Adam(model.parameters(), lr=1e-3)

for xb, yb in batches:
    logits = model(xb)
    loss = criterion(logits, yb)
    model.backward(criterion.backward())   # no autograd: this is the chain rule, by hand
    opt.step()
    opt.zero_grad()
```

## Running it

```bash
pip install -r requirements.txt
pytest -q          # gradient checks, takes well under a second
python train.py    # downloads MNIST (~11 MB) on first run, then trains for 10 epochs on CPU
```

A few knobs if you want to play with it:

```bash
python train.py --optim sgd --lr 0.05       # SGD with momentum instead of Adam
python train.py --hidden 256 --dropout 0.2  # wider hidden layer, with dropout
```

## Results

Default setup: 784-128-10 MLP, ReLU, He init, Adam at lr 1e-3, batch size 64, 10 epochs, no regularization. The whole run takes under a minute on a laptop CPU.

```
epoch  1/10 | loss 0.2432 | train acc 92.66% | test acc 96.11% | 2.8s
epoch  2/10 | loss 0.1081 | train acc 96.76% | test acc 96.99% | 3.5s
epoch  3/10 | loss 0.0744 | train acc 97.67% | test acc 97.31% | 4.1s
epoch  4/10 | loss 0.0569 | train acc 98.22% | test acc 97.40% | 3.6s
epoch  5/10 | loss 0.0472 | train acc 98.49% | test acc 97.57% | 3.9s
epoch  6/10 | loss 0.0387 | train acc 98.75% | test acc 97.41% | 4.6s
epoch  7/10 | loss 0.0302 | train acc 99.02% | test acc 97.62% | 4.6s
epoch  8/10 | loss 0.0269 | train acc 99.10% | test acc 97.46% | 5.4s
epoch  9/10 | loss 0.0238 | train acc 99.20% | test acc 97.46% | 3.7s
epoch 10/10 | loss 0.0185 | train acc 99.37% | test acc 97.38% | 4.2s
Final test accuracy 97.38% (best 97.62%)
```

| | |
|---|---|
| Test accuracy after 10 epochs | **97.38%** |
| Best test accuracy (epoch 7) | **97.62%** |
| Parameters | 101,770 |
| Time per epoch | ~4 s |

You can see the model start to overfit after epoch 5 or so: train accuracy keeps climbing toward 99.4% while test accuracy sits around 97.5%. Adding `--dropout 0.2` or a little weight decay closes that gap. I left the default run unregularized because it makes the overfitting visible in the plot, which I think is more interesting than hiding it.

## The math, briefly

Backprop is just the chain rule applied layer by layer. If $\delta = \partial L / \partial y$ is the gradient flowing in from the layer above, each layer has to produce two things: the gradient for its own parameters, and $\partial L / \partial x$ to hand to the layer below.

**Linear layer**, $y = xW + b$:

$$
\frac{\partial L}{\partial W} = x^\top \delta,\qquad
\frac{\partial L}{\partial b} = \sum_{i} \delta_i,\qquad
\frac{\partial L}{\partial x} = \delta\, W^\top
$$

These three lines are the entire `Linear.backward`. The bias is added once per sample, which is why its gradient is a sum over the batch.

**Element-wise activations**, $y = \sigma(x)$: just $\partial L / \partial x = \delta \odot \sigma'(x)$.

| | $\sigma(x)$ | $\sigma'(x)$ |
|---|---|---|
| ReLU | $\max(0, x)$ | $\mathbb{1}[x > 0]$ |
| Sigmoid | $s = 1/(1+e^{-x})$ | $s(1-s)$ |
| Tanh | $t = \tanh x$ | $1 - t^2$ |

Sigmoid and Tanh derivatives can be written in terms of the output, so the forward pass only needs to cache what it returned.

**Softmax + cross-entropy.** This was the part that made the most sense to me once I saw it. If you differentiate softmax and cross-entropy separately, you get a $C \times C$ Jacobian per sample that is slow and numerically touchy. If you differentiate them together, almost everything cancels and you are left with

$$
\frac{\partial \ell}{\partial z_j} = p_j - \mathbb{1}[j = y]
$$

which is literally "predicted probability minus the one-hot label". That is why the two are fused into one class here.

The other thing softmax needs is the log-sum-exp trick. Computing $e^{z}$ directly overflows for large logits, so instead:

$$
\log p_j = z_j - \max_k z_k - \log \sum_k e^{z_k - \max_k z_k}
$$

Subtracting the max keeps every exponent at or below 1. There is a test that feeds in logits of $[1000, -1000, 0]$ to make sure this does not produce `inf` or `nan`.

**Sequential** is a `for layer in reversed(self.layers)` loop that passes each layer's returned gradient into the previous one. That loop is the chain rule.

**Optimizers.** SGD with momentum keeps a velocity $v \leftarrow \mu v + g$ and steps along it. Adam keeps running averages of the gradient and its square, bias-corrects them, and divides one by the square root of the other.

## How I know the gradients are right

Writing backward passes by hand is easy to get subtly wrong: a missing transpose, a sum over the wrong axis, a factor of $1/N$ that should not be there. Everything still trains, just worse, and you would never know. So every layer is checked against central finite differences:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h},\quad h = 10^{-5}
$$

To turn a layer with a vector output into a scalar function, I dot the output with a fixed random matrix $R$. Then the analytic gradient is simply `layer.backward(R)`, and it has to match the numerical one to a relative error below $10^{-6}$ in float64. There is one check per layer, one for each of $W$, $b$ and $x$ in `Linear`, and one that chains three `Linear` layers, `Tanh`, `ReLU` and the loss together to make sure `Sequential` passes gradients through correctly.

A small detail that bit me: ReLU is not differentiable at exactly 0, so the test nudges any input that lands too close to 0 before checking. Otherwise the finite difference straddles the kink and the test fails for a reason that has nothing to do with the implementation.

## Things I chose not to do

- **No autograd.** A computation graph would be the next step, but the point of this project was to see the derivatives, not hide them.
- **No GPU.** A 100k-parameter MLP in NumPy runs at a few seconds per epoch, which is plenty for MNIST.
- **No convolutions, batch norm, or learning-rate schedules.** All reasonable extensions. The gradient-check harness makes adding a new layer safe, which is part of why I built it first.
