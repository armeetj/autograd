import math
value_error = TypeError("Value type can only hold float scalars")
version = "v0.0"

def parse_value(x):
        if isinstance(x, (int, float)):
            x = Value(float(x))
        elif not isinstance(x, Value):
            raise value_error
        return x
    
class Value:
    def __init__(self, data, _children=()):
        if isinstance(data, (int, float)):
            data = float(data)
        else:
            raise value_error
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __add__(self, other):
        other = parse_value(other)
        res = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        return res
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = parse_value(other)
        res = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad
        res._backward = _backward
        return res
    def __rmul__(self, other):
        return self * other
    def __neg__(self):
        return self * -1
        
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        other = parse_value(other)
        if other.data == 0:
            raise ZeroDivisionError()
        return self * other ** -1
    def __rtruediv__(self, other):
        other = parse_value(other)
        return other / self

    def exp(self):
        res = Value(math.e ** self.data, (self,))
        def _backward():
            self.grad += res.data * res.grad
        res._backward = _backward
        return res

    def sigmoid(self):
        pass
        
    def tanh(self):
        pass

    def relu(self):
        res = Value(max(0, self.data), (self,))
        def _backward():
            if self.data > 0:
                self.grad += res.grad
        res._backward = _backward
        return res

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError("Value type can only be raised to int/float powers")
        res = Value(self.data ** other, (self,))
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * res.grad
        res._backward = _backward
        return res
    def __rpow__(self, other):
        raise NotImplementedError("Value type can only be raised to int/float powers")

    def zero_grad(self):
        self.grad = 0.0
        
    def backward(self):
        _ordered = list()
        _visited = set()
        def add_list(curr):
            if curr not in _visited:
                _visited.add(curr)
                for p in curr._prev:
                    add_list(p)
                _ordered.append(curr)
        add_list(self)
        self.grad = 1.0
        for value in reversed(_ordered):
            value._backward()

    def __float__(self):
        return self.data
        
    def __repr__(self):
        return f"Value({self.data}, grad={self.grad})"