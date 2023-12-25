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
model = nn.Net(3, [10, 1])
print(model)
print(f"trainable_params={model.nparams()}")

lr = 0.01
epochs = 5000

pbar = tqdm.trange(1, epochs + 1, desc="[loss: xxxxxx]")
for epoch in pbar:
    loss = 0
    # forward
    for x, yi in zip(X, y):
        loss += (model(x) - yi) ** 2
    pbar.set_description(f"[loss={str(loss.data)[:6]}][epoch={epoch}]")
    # backward
    model.zero_grad()
    loss.backward()

    # update params
    # does same thing as model.step(lr)
    for p in model.params():
        p.data -= lr * p.grad

print("y_actual:", y)
print("y_pred(class):",[round(model(x).data) for x in X])
print("y_pred:", [model(x).data for x in X])