# armeetjatyani/autograd

Autograd engine and neural network library with PyTorch-like API. Recreation of [micrograd](https://github.com/karpathy/micrograd).

## autograd engine

`autograd/engine.py`

We can construct expressions using the `Value` class. Internally, the library maintains a graph representation of variables and their dependencies.

```python
from autograd.engine import Value
x = Value(2)
y = Value(2.3)
print(x / y) # Value(0.8695652173913044, grad=0.0)

# simple expressions
print(2 + x) # Value(4.0, grad=0.0)
print(x / 0) # ZeroDivisionError
print(x ** 3) # Value(8.0, grad=0.0)
print(x.exp()) # Value(7.3890560989306495, grad=0.0)

# non-linearities
z = Value(-3)
print(z.relu()) # Value(0.0, grad=0.0)
print(z.sigmoid()) # Value(0.9525741268224331, grad=0.0)
```

In the following example, we'll minimize `(x-y)^3 + e^y` over many iterations. In each step, we perform a forward pass, backward pass (to calculate gradients), and then update the parameters x and y according to their calculated gradients.
```python
import numpy as np
from autograd.engine import Value

# initialize params
x = Value(7)
y = Value(3)
verbose = True
lr = 0.001
epochs = 20000
xs = np.arange(0, epochs + 1, 1)
ys = []

for i in range(epochs + 1):
    # forward
    f = (x - y) ** 3 + y.exp()
    f = f.relu() + y
    ys.append(f.data)

    # backward
    x.zero_grad()
    y.zero_grad()    
    f.backward()

    # update params
    x.data -= lr * x.grad
    y.data -= lr * y.grad
```
The following video shows the value of the expression over thousands of iterations. We expect the value to grow smaller and smaller.

https://github.com/armeetjatyani/autograd/assets/38377327/83555924-02bc-4b61-82d8-c642c33a8f78

## nn library

`autograd/nn.py`
