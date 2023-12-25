import sys, os, tqdm, random, math
import matplotlib.pyplot as plt
import numpy as np

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
noise = lambda: random.uniform(-0.1, 0.1)
f = lambda x: math.sin(x[0])
X = [[random.uniform(-2 * math.pi, 2 * math.pi)] for i in range(50)]
y = [f(x) + noise() for x in X]

# construct model
model = nn.Net(1, [20, 20, 1])
print(model)
print(f"trainable_params={model.nparams()}")

lr = 0.001
epochs = 1000

pbar = tqdm.trange(epochs, desc="[loss: xxxxxx]")
for epoch in pbar:
    # forward
    loss = 0
    for x, yi in zip(X, y):
        loss += (model(x) - yi) ** 2
    pbar.set_description(f"[loss={str(loss.data)[:6]}][epoch={epoch}]")
    
    # backward
    model.zero_grad()
    loss.backward()

    # update params
    model.step(lr)

plt.figure()
X_true = list(np.linspace(-2 * math.pi, 2 * math.pi, 100))
y_true = [math.sin(x) for x in X_true]
plt.plot(X_true, y_true, label="actual", c="gray")
plt.scatter(X_true, [model([x]) for x in X_true], label="predicted")
plt.show()
