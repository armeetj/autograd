import sys, os, tqdm, random, math

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "../..")
autograd_path = os.path.abspath(parent_dir)
sys.path.append(autograd_path)

from autograd.engine import Value
import autograd.nn as nn

""" 
generate a dataset for the target function
f(x) = sin(x_1 * x_2 * x_3) + noise
"""
noise = lambda: random.uniform(-0.3, 0.3)
f = lambda x: math.sin(x[0] + x[1] + x[2])
X = [[random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)] for i in range(100)]
y = [f(x) + noise() for x in X]

# construct model
model = nn.Net(3, [100, 50, 1])
print(model)
print(f"trainable_params={model.nparams()}")

lr = 0.1
epochs = 10

pbar = tqdm.trange(epochs, desc="[loss: xxxxxx]")
for epoch in pbar:
    loss = 0
    # forward
    for x, yi in zip(X, y):
        loss += (model(x)[0] - yi) ** 2
    pbar.set_description(f"[loss={str(loss.data)[:6]}][epoch={epoch}]")
    # backward
    model.zero_grad()
    loss.backward()

    # update params
    model.step(lr)
print([model(x)[0].data for x in X])