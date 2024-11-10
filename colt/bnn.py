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