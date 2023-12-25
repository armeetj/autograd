# armeetjatyani/autograd

Autograd engine and neural network library with PyTorch-like API. Recreation of [micrograd](https://github.com/karpathy/micrograd).

- [armeetjatyani/autograd](#armeetjatyaniautograd)
  - [nn library](#nn-library)
    - [Training (demo)](#training-demo)
    - [Neuron](#neuron)
    - [Layer](#layer)
    - [Net](#net)
  - [autograd engine](#autograd-engine)
    - [Minimize (demo)](#minimize-demo)


## nn library

`autograd/nn.py`

The nn library is built on top of the autograd engine (see below). Read about the `Neuron`, `Layer`, and `Net` objects below. Here is a demo showing how a simple network can be trained. Note that this library is meant as an educational exercise. Everything is running on the CPU. Nothing is parallelized. So, we use a tiny example to show the capabilities of this tiny library.

### Training (demo)
TODO: write demo here

### Neuron

Every neuron stores parameters (weights and bias) as `Value` objects that keep track of gradients. To evaluate a neuron, inputs are multiplied by weights and added to the bias. Finally, the resulting signal is activated by a sigmoid function. This entire expression is internally represented by a DAG, to enable backwards gradient calculation (by `autograd.engine`).

Usage:

```python
from autograd.nn import Neuron
n = Neuron(10)

print(n.nparams())
# 11

print(n.params())
# [Value(-0.8269341980053548, grad=0.0), Value(-0.9198909182804311, grad=0.0), Value(0.23878371951669064, grad=0.0), Value(-0.9616815732362081, grad=0.0), Value(0.7005652557465922, grad=0.0), Value(-0.34538779319877766, grad=0.0), Value(0.8949940702869532, grad=0.0), Value(-0.9398044368005902, grad=0.0), Value(0.009769044293206797, grad=0.0), Value(0.03367950339845449, grad=0.0), Value(0.0, grad=0.0)]

# evaluate the neuron, pass in 10 inputs
print(n([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
# Value(0.9944882835077717, grad=0.0)
```

We initialize this neuron to take in 10 inputs. This neuron has 11 total parameters (10 weights + 1 bias).

### Layer

A Layer is an abstraction built on top of the Neuron class. Layers contain identically shaped neurons. That is, each neuron in the layer takes in the same number of inputs. The number of neurons in the layer is the number of the outputs of this layer.

In summary, a layer takes in `N_in` inputs and returns `N_out` outputs.

Usage:

```python
from autograd.nn import Layer
l = Layer(3, 10)
print(l)
# Layer(N_in=3, N_out=10)

print(l.nparams())
# 40

# evaluate the layer (pass in 3 inputs, get 10 outputs)
print(l([1, 2, 3]))
"""
[Value(0.48653402984121236, grad=0.0),
 Value(0.21033094069723815, grad=0.0),
 Value(0.9697836303741563, grad=0.0),
 Value(0.06583501067003876, grad=0.0),
 Value(0.4700712216294792, grad=0.0),
 Value(0.8908593203427052, grad=0.0),
 Value(0.2655114954121276, grad=0.0),
 Value(0.32645743048687054, grad=0.0),
 Value(0.18878689965997322, grad=0.0),
 Value(0.8124174289699577, grad=0.0)]
"""
```

In this example we build a layer with 10 neurons. Each neuron has 4 trainable parameters (3 weights + 1 bias). This yields a total of 40 trainable parameters.

### Net

Net is a multilayer perceptron built by chaining together multiple Layer objects. Layers are fully connected.

Usage:

```python
net = Net(3, [100, 200, 2])
print(net)
# Net([3, 100, 200, 2]): ['Layer(N_in=3, N_out=100)', 'Layer(N_in=100, N_out=200)', 'Layer(N_in=200, N_out=2)']]

print(net.nparams())
# 21002

print(net([1, 2, 3]))
# [Value(0.9786990460374885, grad=0.0), Value(0.9993422981186754, grad=0.0)]
```

In this example, we create a net that has 3 fully connected layers. The net ultimately takes in 3 inputs and yields 2 outputs (ignoring the hidden layers).

- Layer 1: 3 inputs --> 100 outputs
- Layer 2: 100 inputs --> 200 outputs
- Layer 3: 200 inputs --> 2 outputs

## autograd engine

`autograd/engine.py`

We can construct expressions using the `Value` class. Internally, the library maintains a graph representation of variables and their dependencies. We can calculate all gradients of parameters of an expression by calling `.backward()` on the result of an expression.

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

### Minimize (demo)
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
