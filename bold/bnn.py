# heavily inspired from
# https://github.com/karpathy/micrograd

import random
from engine import Bool
from engine import Value
import numpy as np

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class ProdTron(Module):

    def __init__(self, nin):
        # w in {-1,1}
        self.w = [Bool(2*random.randint(0,1)-1) for _ in range(nin)]

    def __call__(self, x):
        act = np.prod([wi^xi for wi,xi in zip(self.w, x)])
        return act

    def parameters(self):
        return self.w

    def __repr__(self):
        return f"ProdTron({len(self.w)})"


class SumTron(Module):

    def __init__(self, nin,xor=True):
        # w in {-1,1}
        if xor:
            self.w = [Bool(2*random.randint(0,1)-1) for _ in range(nin)]
        else:
            self.w = [Bool(1) for _ in range(nin)]

    def __call__(self, x):
        act = np.sum([wi^xi for wi,xi in zip(self.w, x)])
        return act

    def parameters(self):
        return self.w

    def __repr__(self):
        return f"SumTron({len(self.w)})"

class ProdLayer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [ProdTron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        #return out[0] if len(out) == 1 else out
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Prod Layer of [{', '.join(str(n) for n in self.neurons)}]"

class SumLayer(Module):

    def __init__(self, nin, nout, **kwargs):
        # first neuron is plain OR 
        self.neurons = [SumTron(nin, xor=(i!=0),**kwargs) for i in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        #return out[0] if len(out) == 1 else out
        return out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Prod Layer of [{', '.join(str(n) for n in self.neurons)}]"

class Layer(Module):

    def __init__(self, nin, nh, nout, **kwargs):
        self.nin = nin
        self.nh = nh
        self.nout = nout
        self.prod = ProdLayer(nin,nh,**kwargs)
        self.sum = SumLayer(nh,nout,**kwargs)

    def __call__(self, x):
        x = self.prod(x)
        out = self.sum(x)
        return out[0] if len(out) == 1 else out

    def parameters(self):
        w = self.prod.parameters()
        w.extend(self.sum.parameters())
        return w

    def __repr__(self):
        return f"Bool Layer of {self.nin} i/ps; {self.nh} sop terms; {self.nout} o/ps."

class MLP(Module):

    def __init__(self, nin, nouts, nh=None):
        if nh is None:
            nh = [10] * len(nouts)

        sz = [nin] + nouts
        self.layers = [Layer(sz[i], nh[i], sz[i+1])  for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"