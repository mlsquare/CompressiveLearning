import torch

def H(input,atzero=torch.tensor(0.0)):
    return torch.heaviside(input, atzero)

class SimpleBOLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1):
        super(SimpleBOLD, self).__init__(params, defaults={"lr": lr})

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Update the parameters
                w = p.data
                dw = p.grad.data
                dw = 2*w*H(w*dw)
                if torch.sum(dw) >0.5:
                    print('flipped')
                p.data.add_(-group['lr']*dw)
                