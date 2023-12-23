import sys, os
sys.path.append("/Users/armeetjatyani/Developer/autograd/")
print(sys.path)
import autograd.nn as nn # custom nn library
import autograd.engine as grad # custom autograd library
import numpy as np

# load dataset
ds = np.load("mnist.npz")
X_train = np.array(ds["train"].T, float)
y_train = np.array(ds["train_labels"][0], int)
X_test = np.array(ds["test"].T, float)
y_test = np.array(ds["test_labels"][0], int)

def one_hot_encode(x):
    enc = np.zeros(10)
    enc[x] = 1
    return enc

def one_hot_decode(enc):
    x = np.argmax(enc)
    return x

X_train /= 255.0
X_test /= 255.0
y_train = np.array([one_hot_encode(y) for y in y_train])
y_test = np.array([one_hot_encode(y) for y in y_test])

# construct model
f = nn.MLP(28 * 28, [100, 50, 10])

# train model
epochs = 10
verbose = True
lr = 0.01
from tqdm import tqdm
for epoch in tqdm(range(epochs), desc="[epoch]"):
    # forward pass
    loss = grad.Value(0)
    for i in tqdm(range(len(X_train[:100])), leave=True, position=1):
        x = X_train[i]
        y_pred = np.array(f(x))
        y_actual = np.array(y_train[i])
        # print(y_pred, y_actual)
        loss = y_pred - y_actual
        # print(loss)
        loss = sum([x ** 2 for x in loss])
        loss = loss ** 0.5

    # backward pass
    loss.backward()

    # update params
    for p in f.params():
        p.data -= lr * p.grad
        p.zero_grad()

    if verbose:
        print(print("\tLOSS:", loss.data)
        