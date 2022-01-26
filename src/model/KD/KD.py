import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T, beta=1):
        super(DistillKL, self).__init__()
        print('Distill: KD beta={} T={}'.format(beta, T))
        self.T = T
        self.beta = beta

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        loss = self.beta * loss
        return loss
