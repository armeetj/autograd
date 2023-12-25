from autograd.engine import Value
import random

class Module:

    def nparams(self):
        return len(self.params())
    def params(self):
        return []
        
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

class Neuron(Module):
    
    def signal(self, x):
        return self.b + sum([self.w[i] * x[i] for i in range(self.N)])
        
    def activate(self, x):
        # return self.signal(x)
        return self.signal(x).sigmoid()
        
    def __init__(self, N):
        self.N = N
        self.w = [Value(random.uniform(-1, 1)) for _ in range(N)]
        self.b = Value(0)

    def params(self):
        return self.w + [self.b]
        
    def __call__(self, x):
        if len(x) != self.N:
            raise Exception(f"Expected list of length: {self.N}")
        return self.activate(x)

    def __repr__(self):
        return f"Neuron({self.N})"

class Layer(Module):

    def __init__(self, N_in, N_out):
        self.N_in = N_in
        self.N_out = N_out
        self.neurons = [Neuron(N_in) for _ in range(N_out)]

    def __call__(self, x):
        assert len(x) == self.N_in
        out = [n(x) for n in self.neurons]
        return out

    def params(self):
        return [p for n in self.neurons for p in n.params()]

    def __repr__(self):
        return f"Layer(N_in={self.N_in}, N_out={self.N_out})"

class Net(Module):

    def __init__(self, N_in, N_outs):
        self.N = [N_in] + N_outs
        self.layers = [Layer(self.N[i], self.N[i+1]) for i in range(len(N_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def __repr__(self):
        # s = str(l) for l in self.layers
        return f"MLP({self.N}): {[str(l) for l in self.layers]}]"
        
        