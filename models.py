import numpy as np
np.random.seed(211)


def softmax(x):
    """x shape: (N, C)"""
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
    
def softmax_gradient(x):
    """x shape: (N, C)"""
    softmax_x = softmax(x)
    return softmax_x * (1 - softmax_x)


class MLP2():

    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)
        self.ctx = []
    
    def zero_grad(self):
        pass
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, self.W2) + self.b2
        out = softmax(z2)
        self.ctx = [x, z1, a1, z2]
        return out
    
    def loss(self, x, y, loss_fn):
        y_pred = self.forward(x)
        x, z1, a1, z2 = self.ctx
        self.ctx = [x, z1, a1, z2, y_pred, y, loss_fn]
        return loss_fn.forward(y_pred, y)
    
    def gradient(self):
        x, z1, a1, z2, y_pred, y, loss_fn = self.ctx
        # self.ctx = []
        grad_out = loss_fn.gradient(y_pred, y)
        grad_z2 = grad_out * softmax_gradient(z2)
        grad_W2 = np.dot(a1.T, grad_z2)
        grad_b2 = np.sum(grad_z2, axis=0)
        grad_a1 = np.dot(grad_z2, self.W2.T) * (z1 > 0)
        grad_z1 = grad_a1
        grad_W1 = np.dot(x.T, grad_z1)
        grad_b1 = np.sum(grad_z1, axis=0)
        return grad_W1, grad_b1, grad_W2, grad_b2

    def hessian(self):
        x, z1, a1, z2, y_pred, y, loss_fn = self.ctx
        #TODO
        ...
        return hess_W1, hess_b1, hess_W2, hess_b2
    
    def parameters(self):
        return self.W1, self.b1, self.W2, self.b2
    
    def set_params(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
