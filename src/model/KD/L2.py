import torch
import torch.nn as nn

class L2(nn.Module):
    def __init__(self, beta=1):
        super(L2, self).__init__()
        print('Distill: L2 beta={}'.format(beta))
        self.beta = beta
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, f_s, f_t):
        return self.loss(f_s, f_t) * self.beta
