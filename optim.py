import torch
torch.manual_seed(211)


def line_search(model, data, labels, loss_fn, p_list, alpha=0.8, beta=0.5):
    params = model.parameters()
    loss0 = model.loss(data, labels, loss_fn)
    dparams = model.gradient()
    while True:
        params_list = [params[i] + alpha*p_list[i] for i in range(len(params))]
        model.set_params(params_list)
        new_loss = model.loss(data, labels, loss_fn)
        sum = 0
        for i in range(len(params)):
            sum += torch.dot(dparams[i].flatten(), p_list[i].flatten())
        if new_loss > loss0 + alpha*sum:
            alpha *= beta
        else:
            break
    model.set_params(params)
    return alpha


# class LBFGS():
#     def __init__(self, model, loss_fn, gtol=1e-5, m=10):
#         self.model = model
#         self.loss_fn = loss_fn
#         self.gtol = gtol
#         self.m = m
#         self.early_stop = False
        
#         num = 0
#         for param in model.parameters():
#             num += param.numel()
#         self.H = torch.eye(num, device=param.device)
#         self.s_list = []
#         self.y_list = []
#         self.rho_list = []
        
#     def step(self, data, labels, epoch):
#         loss = self.model.loss(data, labels, self.loss_fn)
#         dparams = self.model.gradient()
#         dparams_list = [dparam.flatten() for dparam in dparams]
#         dparams_flatten = torch.cat(dparams_list)
        
#         # check if the gradient is small enough
#         if torch.norm(dparams_flatten) < self.gtol:
#             self.early_stop = True
#             print('epoch {} is converged'.format(epoch))
#         else:
#             # Compute the search direction using L-BFGS formula
#             q = dparams_flatten.clone().detach()
#             alpha_list = []
#             for i in range(len(self.s_list)-1, -1, -1):
#                 rho = self.rho_list[i]
#                 s = self.s_list[i]
#                 y = self.y_list[i]
#                 alpha = rho * torch.dot(s, q) 
#                 alpha_list.append(alpha)
#                 q -= alpha * y
            
#             H0 = torch.dot(self.s_list[-1], self.y_list[-1]) / torch.dot(self.y_list[-1], self.y_list[-1])
#             r = H0 * q
#             for i in range(len(self.s_list)):
#                 rho = self.rho_list[-i-1]
#                 s = self.s_list[-i-1]
#                 y = self.y_list[-i-1]
#                 beta = rho * torch.dot(y, r)
#                 r += s * (alpha_list[-i-1] - beta)
#             p = -r
            
#             num_dparams_list = [dparam.numel() for dparam in dparams]
#             p_list = torch.split(p, num_dparams_list)
#             p_list = [p.reshape(dparam.shape) for p, dparam in zip(p_list, dparams)]
            
#             # Compute the step size using the line search algorithm
#             alpha = line_search(self.model, data, labels, self.loss_fn, p_list=p_list, alpha=1, beta=0.5)
            
#             # Compute the new parameters and the new gradient
#             params = self.model.parameters()
#             params_list = [params[i] + alpha*p_list[i] for i in range(len(params))]
#             self.model.set_params(params_list)
#             loss = self.model.loss(data, labels, self.loss_fn)
#             dparams_new = self.model.gradient()
            
#             # Update s, y and rho lists
#             s = alpha * p
#             y = torch.cat([(dparams_new[i]-dparams[i]).flatten() for i in range(len(dparams))])
#             rho = 1 / (torch.dot(y, s) + 1e-10)
#             self.s_list.append(s)
#             self.y_list.append(y)
#             self.rho_list.append(rho)
            
#             # Check if the size of s, y and rho lists exceeds m, and remove the oldest element
#             if len(self.s_list) > self.m:
#                 self.s_list.pop(0)
#                 self.y_list.pop(0)
#                 self.rho_list.pop(0)
            
#             # Compute the search direction using L-BFGS recursion
#             q = dparams_flatten.clone().detach()
#             alpha_list = []
#             for i in reversed(range(len(self.s_list))):
#                 s_i, y_i, rho_i = self.s_list[i], self.y_list[i], self.rho_list[i]
#                 alpha_i = rho_i * torch.dot(s_i, q)
#                 q = q - alpha_i * y_i
            
#             r = self.H0 @ q
#             for i in range(len(self.s_list)):
#                 s_i, y_i, rho_i = self.s_list[i], self.y_list[i], self.rho_list[i]
#                 beta_i = rho_i * torch.dot(y_i, r)
#                 r = r + (alpha_list[-(i+1)] - beta_i) * s_i
            
#             # Split the search direction into a list of tensors
#             p_list = torch.split(r, num_dparams_list)
#             p_list = [p.reshape(dparam.shape) for p, dparam in zip(p_list, dparams)]
            
#             # Compute the step size using the line search algorithm
#             alpha = line_search(self.model, data, labels, self.loss_fn, p_list=p_list, alpha=1, beta=0.5)
            
#             # Compute the new parameters and the new gradient
#             params = self.model.parameters()
#             params_list = [params[i] + alpha*p_list[i] for i in range(len(params))]
#             self.model.set_params(params_list)
#             loss = self.model.loss(data, labels, self.loss_fn)
#             dparams_new = self.model.gradient()
            
#             # Compute the approximation of the inverse Hessian matrix using L-BFGS recursion
#             s_new = alpha * p
#             y_new = torch.concatenate([(dparams_new[i]-dparams[i]).flatten() for i in range(len(dparams))])
#             rho_new = 1 / (torch.dot(y_new, s_new) + 1e-10)
#             self.s_list.append(s_new)
#             self.y_list.append(y_new)
#             self.rho_list.append(rho_new)
#             dparams_new_flatten = torch.concatenate([dparam_new.flatten() for dparam_new in dparams_new])
#             q = dparams_new_flatten.clone().detach()
#             alpha_list = []
#             for i in range(len(self.s_list)):
#                 s_i, y_i, rho_i = self.s_list[i], self.y_list[i], self.rho_list[i]
#                 alpha_i = rho_i * torch.dot(y_i, q)
#                 alpha_list.append(alpha_i)
#                 q = q - alpha_i * s_i
            
#             r = self.H0 @ q
#             for i in reversed(range(len(self.s_list))):
#                 s_i, y_i, rho_i = self.s_list[i], self.y_list[i], self.rho_list[i]
#                 beta_i = rho_i * torch.dot(s_i, r)
#                 r = r + (alpha_list[-(i+1)] - beta_i) * y_i
            
#             H_new = r @ q.T / torch.dot(y_new, y_new)  # A rank-1 update
            
#             # Update the Hessian matrix
#             self.H0 = H_new
            
#             return self.model


class LBFGS():
    def __init__(self, model, loss_fn, gtol=1e-5, max_iter=100, max_eval=100, M=100):
        self.model = model
        self.loss_fn = loss_fn
        self.gtol = gtol
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.M = M
        self.early_stop = False

    def step(self, data, labels, epoch):
        # Initial loss and gradient
        loss = self.model.loss(data, labels, self.loss_fn)
        dparams = self.model.gradient()
        dparams_list = [dparam.flatten() for dparam in dparams]
        dparams_flatten = torch.cat(dparams_list)
        
        # Initialize H0 to identity matrix
        num = 0
        for param in self.model.parameters():
            num += param.numel()
        self.H0 = torch.eye(num, device=param.device)  
        self.old_dirs = []
        self.old_stps = []

        # Check if gradient is small, if so stop
        if torch.norm(dparams_flatten) < self.gtol:
            self.early_stop = True
            print('L-BFGS converged at epoch {}'.format(epoch))
        
        # Compute search direction
        Hd = self.H0 @ dparams_flatten
        for m in range(min(self.M, len(self.old_dirs))):
            q = -self.old_stps[m] @ dparams_flatten 
            r = self.old_dirs[m] @ dparams_flatten  
            alpha = (r + self.old_stps[m] @ q) / (self.old_stps[m] @ Hd) 
            Hd += -alpha * self.old_dirs[m]

        # Perform line search  
        p = -Hd
        num_dparams_list = [dparam.numel() for dparam in dparams]
        p_list = torch.split(p, num_dparams_list)
        p_list = [p.reshape(dparam.shape) for p, dparam in zip(p_list, dparams)]
        alpha = line_search(self.model, data, labels, self.loss_fn, p_list, alpha=1, beta=0.5)
        
        # Update params and evaluate objective 
        params = self.model.parameters()
        params_list = [params[i] + alpha*p_list[i] for i in range(len(params))]
        self.model.set_params(params_list)
        loss_new = self.model.loss(data, labels, self.loss_fn)
        
        # Save info for next iteration of L-BFGS 
        self.old_dirs.insert(0, dparams_flatten.clone())
        self.old_stps.insert(0, alpha*p.clone())
        if len(self.old_dirs) > self.M:
            self.old_dirs.pop()
            self.old_stps.pop()
            
        # Update gradient
        dparams_new = self.model.gradient()          
        dparams_list = [dparam_new.flatten() for dparam_new in dparams_new]
        dparams_flatten = torch.cat(dparams_list) 
            
        return self.model


class BFGS():
    def __init__(self, model, loss_fn, gtol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.gtol = gtol
        self.early_stop = False
        num = 0
        for param in model.parameters():
            num += param.numel()
        self.H = torch.eye(num, device=param.device)

    def step(self, data, labels, epoch):
        loss = self.model.loss(data, labels, self.loss_fn)
        dparams = self.model.gradient()
        dparams_list = [dparam.flatten() for dparam in dparams]
        dparams_flatten = torch.concatenate(dparams_list)
        # check if the gradient is small enough
        if torch.norm(dparams_flatten) < self.gtol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            self.H = self.H
            # Compute the search direction using the BFGS formula
            p = -self.H @ dparams_flatten
            num_dparams_list = [dparam.numel() for dparam in dparams]
            p_list = torch.split(p, num_dparams_list)
            p_list = [p.reshape(dparam.shape) for p, dparam in zip(p_list, dparams)]

            # Compute the step size using the line search algorithm
            alpha = line_search(self.model, data, labels, self.loss_fn, p_list=p_list, alpha=1, beta=0.5)
            
            # Compute the new parameters and the new gradient
            params = self.model.parameters()
            params_list = [params[i] + alpha*p_list[i] for i in range(len(params))]
            self.model.set_params(params_list)
            loss = self.model.loss(data, labels, self.loss_fn)
            dparams_new = self.model.gradient()
            
            # Compute the approximation of the inverse Hessian matrix using the BFGS formula
            s = alpha * p
            y = torch.concatenate([(dparams_new[i]-dparams[i]).flatten() for i in range(len(dparams))])
            rho = 1 / (torch.dot(y, s) + 1e-10)
            I = torch.eye(s.numel(), device=y.device)
            s = s.reshape(-1, 1); y = y.reshape(-1, 1)
            H_new = (I - rho * s @ y.T) @ self.H @ (I - rho * y @ s.T) + rho * s @ s.T
            
            # Update the Hessian matrix
            self.H = H_new

        return self.model



class Newton():
    ...



class GD():
    def __init__(self, model, loss_fn, lr=1e-3, gtol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.gtol = gtol
        self.early_stop = False

    def step(self, data, labels, epoch):
        loss = self.model.loss(data, labels, self.loss_fn)
        dparams = self.model.gradient()
        dparams_list = [dparam.flatten() for dparam in dparams]
        if torch.norm(torch.concatenate(dparams_list)) < self.gtol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            params_list = []
            for param, dparam in zip(self.model.parameters(), dparams):
                param -= self.lr * dparam
                params_list.append(param)
            self.model.set_params(params_list)
        return self.model


class SD():
    def __init__(self, model, loss_fn, gtol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.gtol = gtol
        self.early_stop = False
    
    def step(self, data, labels, epoch):
        loss = self.model.loss(data, labels, self.loss_fn)
        dparams = self.model.gradient()
        dparams_list = [dparam.flatten() for dparam in dparams]
        if torch.norm(torch.concatenate(dparams_list)) < self.gtol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            # Compute steepest descent directions
            p_list = [-1*dparam for dparam in dparams]
            
            # Line search to determine optimal step size
            alpha = line_search(self.model, data, labels, self.loss_fn, p_list, alpha=1, beta=0.5) 
            
            params_list = []
            for param, p in zip(self.model.parameters(), p_list):
                param += alpha * p
                params_list.append(param)
            self.model.set_params(params_list)
 
        return self.model