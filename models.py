import torch
torch.manual_seed(211)


# def softmax_gradient(x):
#     """x shape: (N, C)"""
#     softmax_x = torch.softmax(x, dim=1)
#     return softmax_x * (1 - softmax_x)

class Linear():

    def __init__(self, input_size, output_size, device='cpu'):
        self.W = torch.randn(input_size, output_size).requires_grad_(True).to(device)
        self.b = torch.randn(output_size).requires_grad_(True).to(device)
        self.ctx = []
    
    def zero_grad(self):
        pass
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        self.ctx = [x]
        z = torch.matmul(x, self.W) + self.b
        out = torch.softmax(z, dim=1)
        return out
    
    def loss(self, x, y, loss_fn):
        y_pred = self.forward(x)
        self.ctx = [x, y_pred, y, loss_fn]
        return loss_fn.forward(y_pred, y)
    
    def gradient(self):
        x, y_pred, y, loss_fn = self.ctx
        grad_W, grad_b = torch.autograd.grad(
            self.loss(x,y,loss_fn), [self.W, self.b])
        return grad_W, grad_b
    
    def parameters(self):
        return self.W, self.b
    
    def set_params(self, params: tuple or list):
        self.W, self.b = params


class MLP2():

    def __init__(self, input_size, hidden_size, output_size, device='cpu'):
        self.W1 = torch.randn(input_size, hidden_size).requires_grad_(True).to(device)
        self.b1 = torch.randn(hidden_size).requires_grad_(True).to(device)
        self.W2 = torch.randn(hidden_size, output_size).requires_grad_(True).to(device)
        self.b2 = torch.randn(output_size).requires_grad_(True).to(device)
        self.ctx = []
    
    def zero_grad(self):
        pass
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        z1 = torch.matmul(x, self.W1) + self.b1
        a1 = torch.relu(z1)
        z2 = torch.matmul(a1, self.W2) + self.b2
        out = torch.softmax(z2, dim=1)
        self.ctx = [x, z1, a1, z2]
        return out
    
    def loss(self, x, y, loss_fn):
        y_pred = self.forward(x)
        x, z1, a1, z2 = self.ctx
        self.ctx = [x, z1, a1, z2, y_pred, y, loss_fn]
        return loss_fn.forward(y_pred, y)
    
    def gradient(self):
        x, z1, a1, z2, y_pred, y, loss_fn = self.ctx
        grad_W1, grad_b1, grad_W2, grad_b2 = torch.autograd.grad(
            self.loss(x,y,loss_fn), [self.W1, self.b1, self.W2, self.b2])
        return grad_W1, grad_b1, grad_W2, grad_b2
    
    def parameters(self):
        return self.W1, self.b1, self.W2, self.b2
    
    def set_params(self, params: tuple or list):
        self.W1, self.b1, self.W2, self.b2 = params