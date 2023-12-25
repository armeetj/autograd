import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")
autograd_path = os.path.abspath(parent_dir)
sys.path.append(autograd_path)

from autograd.engine import Value
import autograd.nn as nn
import tqdm

X = [[1, 2, 3], [2, 3, 4], [-3, 4, -2.3], [-5, 6, -2]]
y = [0, 1, 0, 1]

# construct model
model = nn.Net(3, [10, 10, 1])

lr = 0.01
epochs = 100

loss = Value(0)
for epoch in tqdm.trange(epochs, postfix=f"loss: {loss.data}"):
    # forward
    for x, yi in zip(X, y):
        loss += (model(x)[0] - yi) ** 2
    loss.zero_grad()
    # backward
    loss.backward()

    # update params
    for p in model.params():
        p.data -= lr * p.grad
