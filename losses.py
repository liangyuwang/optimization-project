import torch
torch.manual_seed(211)


class cross_entropy():
    def __init__(self):
        pass
    def forward(self, y_pred, y):
        # torch.finfo(torch.float16).eps
        return -torch.sum(y * torch.log(y_pred + 1e-8))
    def gradient(self, y_pred, y):
        return -y / (y_pred + 1e-8)