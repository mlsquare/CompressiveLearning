import torch
import torch.nn as nn
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import laplace  # Import Laplace distribution

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

def _SAND(input,weight):
    # ops are not recorded by autograd
    eps = 0.1
    atzero = torch.tensor([0.0])
    x = input.clone().detach()
    w = weight.clone().detach()
    
    h1 = torch.sum(0.5*(w+1))-eps
    h1 = torch.heaviside(h1,atzero)
    
    h2 = (0.25*(x-1)*(w+1))
    h2 = h2.sum(-1)+eps
    h2 = 2*torch.heaviside(h2, atzero)-1
        
    output = h1*h2
    return output

# Function to apply scalar_function to each row, excluding one feature at a time
def _Apply_Scaler_Function(tensor_x, tensor_w, scalar_function):
    n = len(tensor_w)
    results = []
    for i in range(n):
        x_new = torch.cat((tensor_x[0:i], tensor_x[i+1:]))  # Exclude the i-th feature
        w_new = torch.cat((tensor_w[0:i], tensor_w[i+1:]))  # Exclude the i-th feature
        results.append(scalar_function(x_new,w_new))
    return torch.tensor(results)



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
            local_weight_grad = torch.zeros(batch_size,input_size)

            for b in range(batch_size):
                x = X[b,:]
                local_Hni = _Apply_Scaler_Function(x,w,_SAND)
                Hni[b,:] = local_Hni 
                local_weight_grad[b,:] = torch.heaviside(x,atzero)*torch.heaviside(local_Hni,atzero)
        
        
        if ctx.needs_input_grad[0]:
            local_input_grad = torch.zeros(batch_size,input_size)
            grad_input = torch.zeros(batch_size,input_size)
            for b in range(batch_size):
                # compute local gradient
                local_input_grad[b,:] = -torch.heaviside(w,atzero)*local_weight_grad[b,:]
                # apply chain rule
            grad_input = local_input_grad*grad_output
    
        
        if ctx.needs_input_grad[1]:
            grad_weight = local_weight_grad*grad_output


        return grad_input, grad_weight
    

class SAND(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features    
        self.weight = nn.Parameter(2 * torch.randint(0, 2, (input_features,)).float() - 1)
        
        
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return SANDFunction.apply(input, self.weight)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}'.format(
            self.input_features)    


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