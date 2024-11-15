import torch
import torch.nn as nn
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import laplace  # Import Laplace distribution

def H(input,atzero=torch.tensor(0.0)):
    return torch.heaviside(input, atzero)

# Function to apply scalar_function to each row, excluding one feature at a time
def _Apply_Scaler_Function(tensor_x, tensor_w, scalar_function):
    n = len(tensor_w)
    results = []
    for i in range(n):
        x_new = torch.cat((tensor_x[0:i], tensor_x[i+1:]))  # Exclude the i-th feature
        w_new = torch.cat((tensor_w[0:i], tensor_w[i+1:]))  # Exclude the i-th feature
        results.append(scalar_function(x_new,w_new))
    return torch.tensor(results)

def _SOR(input,weight):
    # ops are not recorded by autograd
    eps = 0.1
    
    x = input.clone().detach()
    w = weight.clone().detach()
    
    h1 = torch.sum(0.5*(w+1))-eps
    h1 = H(h1)
    
    h2 = (0.25*(x+1)*(w+1))
    h2 = h2.sum(-1)-eps
    h2 = 2*H(h2)-1
        
    output = h1*h2

    if output.dim() == 0:  # Scalar tensor
        output = output.view(1, 1)
    else:
        output = output.view(len(output), 1)

    return output

def _SAND(input,weight):
    # ops are not recorded by autograd
    eps = 0.1
    atzero = torch.tensor([0.0])
    
    x = input.clone().detach()
    w = weight.clone().detach()
    
    h1 = torch.sum(0.5*(w+1))-eps
    h1 = H(h1)
    
    h2 = (0.25*(x-1)*(w+1))
    h2 = h2.sum(-1)+eps
    h2 = 2*H(h2)-1

    output = h1*h2
    
    if output.dim() == 0:  # Scalar tensor
        output = output.view(1, 1)
    else:
        output = output.view(len(output), 1)


    return output

def _SXOR(input,weight):
    # ops are not recorded by autograd
    eps = 0.1
    atzero = torch.tensor([0.0])
    
    x = input.clone().detach()
    w = weight.clone().detach()
    
    h1 = torch.sum(0.5*(w+1))-eps
    h1 = H(h1)
    
    h2 = (0.25*(x-1)*(w+1))
    h2 = h2.sum(-1)+eps
    h2 = 2*torch.remainder(h2, 2)-1

    output = h1*h2
    
    if output.dim() == 0:  # Scalar tensor
        output = output.view(1, 1)
    else:
        output = output.view(len(output), 1)


    return output


# Inherit from Function
# https://pytorch.org/docs/stable/notes/extending.html
class FlipFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight):
        output = -input*weight
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)

    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        
        if ctx.needs_input_grad[0]:
            local_grad = -weight
            grad_input = torch.sign(grad_output*local_grad)
        
        
        if ctx.needs_input_grad[1]:
            local_grad = -input
            grad_weight = torch.sign(grad_output*local_grad)
        
        
        return grad_input, grad_weight

class Flip(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features    
        self.weight = nn.Parameter(2 * torch.randint(0, 2, (input_features,)).float() - 1)
        
        
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return FlipFunction.apply(input, self.weight)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}'.format(
            self.input_features)    






# https://pytorch.org/docs/stable/notes/extending.html
class SANDFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight):
        output = _SAND(input,weight)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)

    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        atzero = torch.tensor([0.0])

        # gradients wrt to weights are needed
        # dw is computed either dw or dw is needed
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if input.dim() == 1:
                input_size = input.shape[0]
                batch_size = 1
            else:
                batch_size = input.shape[0]
                input_size = input.shape[1]
            
            X = input.clone().detach().view(batch_size,input_size)
            w = weight.clone().detach()
            Hni = torch.zeros(batch_size,input_size)
        
            for b in range(batch_size):
                x = X[b,:]
                Hni[b,:] = _Apply_Scaler_Function(x,w,_SAND)
                
        
        
        if ctx.needs_input_grad[0]:
            local_input_grad = torch.zeros(batch_size,input_size)
            grad_input = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                # compute local gradient
                x = X[b,:]
                local_Hni = Hni[b,:]
                local_input_grad[b,:] = -H(w)*H(x)*H(local_Hni)
                # apply chain rule
            local_input_grad = 0.2+0.8*local_input_grad
            grad_input = local_input_grad*grad_output
    
        
        if ctx.needs_input_grad[1]:
            local_weight_grad = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                x = X[b,:]
                local_Hni = Hni[b,:]
                local_weight_grad[b,:] = H(x)*H(local_Hni)
            local_weight_grad = 0.2+0.8*local_weight_grad
            grad_weight = local_weight_grad*grad_output


        return grad_input, grad_weight
    

class SAND(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features    
        #self.weight = nn.Parameter(2 * torch.randint(0, 2, (input_features,)).float() - 1)
        self.weight = nn.Parameter(1.0*torch.ones(input_features,))
        
        
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SANDFunction.apply(input, self.weight)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}'.format(
            self.input_features)    


class SORFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight):
        output = _SOR(input,weight)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)

    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        atzero = torch.tensor([0.0])
        eps = 0.1

        # gradients wrt to weights are needed
        # dw is computed either dw or dw is needed
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if input.dim() == 1:
                input_size = input.shape[0]
                batch_size = 1
            else:
                batch_size = input.shape[0]
                input_size = input.shape[1]
            
            X = input.clone().detach().view(batch_size,input_size)
            w = weight.clone().detach()
            Hni = torch.zeros(batch_size,input_size)
            

            for b in range(batch_size):
                x = X[b,:]
                Hni[b,:] = _Apply_Scaler_Function(x,w,_SOR)
        
        if ctx.needs_input_grad[0]:
            local_input_grad = torch.zeros(batch_size,input_size)
            grad_input = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                # compute local gradient
                x = X[b,:]
                local_input_grad[b,:] = H(w)*H(-x+eps)
                # apply chain rule
            local_input_grad = 0.2+0.8*local_input_grad
            grad_input = local_input_grad*grad_output
    
        
        if ctx.needs_input_grad[1]:
            local_weight_grad = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                x = X[b,:]
                local_Hni = Hni[b,:]
                local_weight_grad[b,:] = H(w)*H(-local_Hni)*H(x)

            local_weight_grad = 0.2+0.8*local_weight_grad
            grad_weight = local_weight_grad*grad_output


        return grad_input, grad_weight
    
class SOR(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features    
        #self.weight = nn.Parameter(2 * torch.randint(0, 2, (input_features,)).float() - 1)
        self.weight = nn.Parameter(-1*torch.ones(input_features,))
        
        
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SORFunction.apply(input, self.weight)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}'.format(
            self.input_features)    

class SXORFunction(torch.autograd.Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight):
        output = _SXOR(input,weight)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight = inputs
        ctx.save_for_backward(input, weight)

    
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        atzero = torch.tensor([0.0])
        eps = 0.1

        # gradients wrt to weights are needed
        # dw is computed either dw or dw is needed
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if input.dim() == 1:
                input_size = input.shape[0]
                batch_size = 1
            else:
                batch_size = input.shape[0]
                input_size = input.shape[1]
            
            X = input.clone().detach().view(batch_size,input_size)
            w = weight.clone().detach()
            Hni = torch.zeros(batch_size,input_size)
            

            for b in range(batch_size):
                x = X[b,:]
                Hni[b,:] = _Apply_Scaler_Function(x,w,_SXOR)
        
        if ctx.needs_input_grad[0]:
            local_input_grad = torch.zeros(batch_size,input_size)
            grad_input = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                # compute local gradient
                x = X[b,:]
                local_input_grad[b,:] = H(w)*H(-x+eps)
                # apply chain rule
            local_input_grad = 0.2+0.8*local_input_grad
            grad_input = local_input_grad*grad_output
    
        
        if ctx.needs_input_grad[1]:
            local_weight_grad = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                x = X[b,:]
                local_Hni = Hni[b,:]
                local_weight_grad[b,:] = H(-local_Hni)*H(x)

            local_weight_grad = 0.2+0.8*local_weight_grad
            grad_weight = local_weight_grad*grad_output


        return grad_input, grad_weight

class SXOR(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features    
        #self.weight = nn.Parameter(2 * torch.randint(0, 2, (input_features,)).float() - 1)
        self.weight = nn.Parameter(1*torch.ones(input_features,))
        
        
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SXORFunction.apply(input, self.weight)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}'.format(
            self.input_features)   

# define sparse BNN layer
class SparseBNNLayer(nn.Module):

    def __init__(self, nin, nhid, nout, **kwargs):
        super(SparseBNNLayer, self).__init__()
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        
        self.FlipAndSANDGates = nn.ModuleList()
        for _ in range(self.nhid):
            self.FlipAndSANDGates.append(nn.Sequential(Flip(nin),SAND(self.nin)))

        self.SORGates = nn.ModuleList()
        
        for _ in range(self.nout):
            self.SORGates.append(SOR(self.nhid))

    def forward(self,x):
        h = torch.cat([sand(x) for sand in self.FlipAndSANDGates],-1)
        out = torch.cat([sor(h) for sor in self.SORGates],-1)
        return out

# define sparse BNN layer
class SparseBNNLinearLayer(nn.Module):

    def __init__(self, nin, nout, **kwargs):
        super(SparseBNNLinearLayer, self).__init__()
        self.nin = nin
        self.nout = nout
        
        self.FlipAndSXORGates = nn.ModuleList()
        for _ in range(self.nout):
            self.FlipAndSXORGates.append(nn.Sequential(Flip(nin),SXOR(self.nin)))

    def forward(self,x):
        out = torch.cat([sxor(x) for sxor in self.FlipAndSXORGates],-1)
        return out

class VisualizeBNNLayer():
    def __init__(self, layer):
        super().__init__()
        self.nin = layer.nin
        self.nhid = layer.nhid
        self.nout = layer.nout
        self.terms = ['x'+str(i+1) for i in range(self.nin)]
        self.layer = layer

    def get_sand_terms(self,sand_block, terms):
        flip = sand_block[0] 
        w = flip.weight.detach().clone().numpy()
        t1 = terms[0] if w[0] > 0 else terms[0]+"'"
        t2 = terms[1] if w[1] > 0 else terms[1]+"'"
            
        sand = sand_block[1]
        w = (sand.weight.detach().clone().numpy().tolist())        
        p = '' 
        p1 = t1 if w[0] > 0 else p
        p2 = t2 if w[1] > 0 else p
        return p1+p2

    def get_sop_terms(slef,sor_block, terms):
        w = (sor_block.weight.detach().clone().numpy())
        sop_terms = ''
        for i, wi in enumerate(w):
            sop_terms = (sop_terms+'+'+terms[i]) if wi >= 0.0 else sop_terms
        return sop_terms

    def get_truth_table(self):
        prod_terms = []
        sop_terms = []
        for name, layer in self.layer.named_children():
            # get the Prod terms
            if name == "FlipAndSANDGates":
                for sand_block in layer:
                    terms = self.get_sand_terms(sand_block, self.terms)
                    prod_terms.append(terms)
            else:
                for sor_block in layer:
                    print(sor_block)
                    sop_terms.append(self.get_sop_terms(sor_block, prod_terms))
        return prod_terms, sop_terms

x = 2 * torch.randint(0, 2, (5,4,)).float() - 1
#x = torch.randn(5, 4)  # Batch size 5, 4 features
model =  Flip(4)
y = model(x)    
print(y.shape)


n = 4
x = torch.tensor(2 * torch.randint(0, 2, (n,)).float() - 1,requires_grad=True)
w = 2 * torch.randint(0, 2, (n,)).float() - 1
model = SAND(4)
y = model(x)
z = y.mean()
z.backward()


n = 4
x = torch.tensor(2 * torch.randint(0, 2, (4,n,)).float() - 1,requires_grad=True)
w = 2 * torch.randint(0, 2, (n,)).float() - 1
model = SOR(4)
y = model(x)
z = y.mean()
z.backward()


n = 4
x = torch.tensor(2 * torch.randint(0, 2, (n,)).float() - 1)
model = SparseBNNLayer(4,10,1)
y = model(x)
z = y.mean()
z.backward() 

n = 4
x = torch.tensor(2 * torch.randint(0, 2, (n,)).float() - 1)
model = SparseBNNLinearLayer(4,2)
y = model(x)
z = y.mean()
z.backward() 
