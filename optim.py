import torch
torch.manual_seed(211)


def backtrack_linesearch_f(model, data, labels, loss_fn, grad, p, alpha=0.1, beta=0.8, max_iter=100):
    t = 1
    counter = 0
    loss0 = model.loss(data, labels, loss_fn).clone().detach()
    params_save2 = [param.clone().detach().requires_grad_(True) for param in model.parameters()]
    params_save1 = params_trans(params_save2)
    # check admissibility condition
    while True:
        params_ = params_save1 + t * p
        model.set_params(param_inverse(params_, model))
        if model.loss(data, labels, loss_fn) > loss0 + alpha * t * grad.T @ p:
            t *= beta
        else:
            break
        counter += 1
        if counter > max_iter: break
    model.set_params(params_save2)
    return t

def params_trans(params_list):
    return torch.cat([param.flatten() for param in params_list]).reshape(-1, 1)

def param_inverse(params, model):
    num_params_list = [param.numel() for param in model.parameters()]
    params_list = torch.split(params, num_params_list)
    params_list = [param1.reshape(param2.shape) for param1, param2 in zip(params_list, model.parameters())]
    return params_list

def grads_trans(grads_list):
    return torch.cat([grad.flatten() for grad in grads_list]).reshape(-1, 1)

def grad_inverse(grads, model):
    num_params_list = [param.numel() for param in model.parameters()]
    grads_list = torch.split(grads, num_params_list)
    grads_list = [grad.reshape(param.shape) for grad, param in zip(grads_list, model.parameters())]
    return grads_list


class LBFGS():
    
    def __init__(self, model, loss_fn, m=5, tol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.tol = tol
        self.early_stop = False
        self.m = m
        num = 0
        for param in model.parameters():
            num += param.numel()
        self.tol_num = num
        self.B_k_inv = torch.eye(self.tol_num, device=param.device)
        self.s_k = torch.zeros((self.m, self.tol_num), device=param.device)
        self.y_k = torch.zeros((self.m, self.tol_num), device=param.device)
    
    def step(self, data, labels, epoch):
        self.model.loss(data, labels, self.loss_fn)
        grad = grads_trans(self.model.gradient())
        if torch.norm(grad) < self.tol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            p_k = self.find_direction(grad, self.s_k, self.y_k, epoch)
            params_old = params_trans(self.model.parameters()).clone()
            alpha = backtrack_linesearch_f(self.model, data, labels, self.loss_fn, grad, p_k)
            params = params_trans(self.model.parameters())
            params = params + alpha * p_k
            self.model.set_params(param_inverse(params, self.model))
            
            self.s_k[:-1,:] = self.s_k[1:,:].clone().detach()
            self.s_k[-1,:] = (params - params_old).flatten().clone().detach()
            self.y_k[:-1,:] = self.y_k[1:,:].clone().detach()
            self.y_k[-1,:] = (grads_trans(self.model.gradient()) - grad).flatten().clone().detach()
        return self.model
    
    def find_direction(self, g_k, s_k, y_k, k):
        q = g_k.clone().detach().flatten()
        alpha = torch.zeros(self.m)
        if k+1 <= self.m:
            indices = list(range(self.m-1, self.m-k-1, -1))
        else:
            indices = list(reversed(range(self.m)))
        for i in indices:
            alpha[i] = torch.dot(s_k[i], q) / torch.dot(y_k[i], s_k[i])
            q -= alpha[i] * y_k[i]
        r = q
        for i in reversed(indices):
            beta = torch.dot(y_k[i], r) / torch.dot(y_k[i], s_k[i])
            r += (alpha[i] - beta) * s_k[i]
        return -r.reshape_as(g_k)


class BFGS():
    
    def __init__(self, model, loss_fn, tol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.tol = tol
        self.early_stop = False
        num = 0
        for param in model.parameters():
            num += param.numel()
        self.tol_num = num
        self.B_k_inv = torch.eye(self.tol_num, device=param.device)
    
    def step(self, data, labels, epoch):
        self.model.loss(data, labels, self.loss_fn)
        grad = grads_trans(self.model.gradient())
        if torch.norm(grad) < self.tol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            p_k = - self.B_k_inv @ grad
            params_old = params_trans(self.model.parameters()).clone()
            alpha = backtrack_linesearch_f(self.model, data, labels, self.loss_fn, grad, p_k)
            params = params_trans(self.model.parameters())
            params = params + alpha * p_k
            self.model.set_params(param_inverse(params, self.model))
            
            # update B_k_inv
            s_k = (params - params_old).detach()
            y_k = (grads_trans(self.model.gradient()) - grad).detach()
            rho = (1 / (y_k.T @ s_k)).detach()
            A = (torch.eye(self.tol_num, device=params.device) - rho * (s_k @ y_k.T)).detach()
            B = (torch.eye(self.tol_num, device=params.device) - rho * (y_k @ s_k.T)).detach()
            self.B_k_inv = A @ self.B_k_inv @ B + rho * (s_k @ s_k.T)
        return self.model


class Newton():
    #TODO torch does not support hessian auto computing
    ...



class GD():
    
    def __init__(self, model, loss_fn, lr=1e-3, tol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.tol = tol
        self.early_stop = False
    
    def step(self, data, labels, epoch):
        self.model.loss(data, labels, self.loss_fn)
        grad = grads_trans(self.model.gradient())
        if torch.norm(grad) < self.tol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            params = params_trans(self.model.parameters())
            params = params - self.lr * grad
            self.model.set_params(param_inverse(params, self.model))
        return self.model


class SD():
    
    def __init__(self, model, loss_fn, tol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.tol = tol
        self.early_stop = False
    
    def step(self, data, labels, epoch):
        self.model.loss(data, labels, self.loss_fn)
        grad = grads_trans(self.model.gradient())
        if torch.norm(grad) < self.tol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            p = -1 * grad
            alpha = backtrack_linesearch_f(self.model, data, labels, self.loss_fn, grad, p)
            params = params_trans(self.model.parameters())
            params = params + alpha * p
            self.model.set_params(param_inverse(params, self.model))
        return self.model