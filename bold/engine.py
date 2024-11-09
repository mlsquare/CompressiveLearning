# heavily inspired from
# https://github.com/karpathy/micrograd

import numpy as np

class Bool:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0; # gradient accumulation
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        # OR operation
        other = other if isinstance(other, Bool) else Bool(other)
        x1 = self.data
        x2 = other.data
        x = (np.abs(x1*x2)>0)*np.sign(x1+x2+1)
        out = Bool(x, (self, other), '+')

        def _backward():
            # XNOR(g'(f(x)),f'(x))
            # f = OR(x,a)
            # g'(f(x)) = z =  out.grad
            if other.data == 0 or self.data == 0:
                self.grad += 0
                other.grad += 0
            else:
                self_df = np.min([0,other.data])
                other_df = np.min([0,self.data])
                dg = out.grad
                self.grad += np.sign(self_df*dg)
                other.grad += np.sign(other_df*dg)

        out._backward = _backward
        return out

    def __mul__(self, other):
        # AND operation
        other = other if isinstance(other, Bool) else Bool(other)
        x1 = self.data
        x2 = other.data
        x = (np.abs(x1*x2)>0)*np.sign(x1*x2+x1+x2)
        out = Bool(x, (self, other), '*')

        def _backward():
            # XNOR(g'(f(x)),f'(x))
            # f = OR(x,a)
            # g'(f(x)) = z =  out.grad
            if other.data == 0 or self.data == 0:
                self.grad += 0
                other.grad += 0
            else:
                self_df = np.max([0,other.data])
                other_df = np.max([0,self.data])
                dg = out.grad
                self.grad += np.sign(self_df*dg)
                other.grad += np.sign(other_df*dg)

        out._backward = _backward
        return out
    
    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __rxor__(self, other): # other * self
        return self ^ other

    
    def __xor__(self, other):
        # XOR operation
        other = other if isinstance(other, Bool) else Bool(other)
        x1 = self.data
        x2 = other.data
        x = np.sign(-x1*x2)
        out = Bool(x, (self, other), '^')

        def _backward():
            # XNOR(g'(f(x)),f'(x))
            # f = XOR(x,a)
            # f' = not(a)
            # g'(f(x)) = z =  out.grad
            if other.data == 0 or self.data == 0:
                self.grad += 0
                other.grad += 0
            else:
                self_df = np.sign(-other.data)
                other_df = np.sign(-self.data)
                dg = out.grad
                self.grad += np.sign(self_df*dg)
                other.grad += np.sign(other_df*dg)

        out._backward = _backward
        return out
    
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1 # 1 (chain rule see through)
        for v in reversed(topo):
            v._backward()
    
    def __repr__(self):
        return f"data:{self.data}, grad:{self.grad}"

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"