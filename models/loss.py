import torch
import torch.nn as nn

# Class Balanced CrossEntropy Loss
class CBCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y):
        b = (1-torch.sum(y))/torch.sum(torch.ones_like(y))
        loss = torch.sum(-b*torch.log(p)*y-(1-b)*torch.log(1-p)*(1-y))
        return loss