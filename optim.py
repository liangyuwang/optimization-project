import numpy as np
np.random.seed(211)


# def line_search(f, x, p, alpha=0.5, beta=0.5):
#     fx, gx = f(x)
#     while f(x + alpha*p)[0] > fx + alpha*beta*np.dot(gx, p):
#         alpha *= beta
#     return alpha

def line_search(model, data, labels, loss_fn, p_list, alpha=0.8, beta=0.5):
    W1, b1, W2, b2 = model.parameters()
    loss0 = model.loss(data, labels, loss_fn)
    dW1, db1, dW2, db2 = model.gradient()
    p_W1, p_b1, p_W2, p_b2 = p_list
    while True:
        model.set_params(W1 + alpha*p_W1, b1 + alpha*p_b1, W2 + alpha*p_W2, b2 + alpha*p_b2)
        new_loss = model.loss(data, labels, loss_fn)
        if new_loss > loss0 + alpha*np.dot(dW1.flatten(), p_W1.flatten()) \
            + alpha*np.dot(db1.flatten(), p_b1.flatten()) \
                + alpha*np.dot(dW2.flatten(), p_W2.flatten()) \
                    + alpha*np.dot(db2.flatten(), p_b2.flatten()):   
            alpha *= beta
        else:
            break
    model.set_params(W1, b1, W2, b2)
    return alpha


class LBFGS:
    def __init__(self, f, x0, gtol=1e-5, maxiter=100, history_size=10):
        self.f = f
        self.x = x0.copy()
        self.gtol = gtol
        self.maxiter = maxiter
        self.history_size = history_size
        self.H = np.eye(len(x0))
        self.old_dirs = []
        self.old_stps = []
        self.old_fxs = []
        
    def zero_grad(self):
        pass
    
    def step(self):
        fx, gx = self.f(self.x)
        p = -np.dot(self.H, gx)
        alpha = 1.0
        while self.f(self.x + alpha*p)[0] > fx + alpha*self.gtol*np.dot(gx, p):
            alpha *= 0.5
        xprev = self.x.copy()
        self.x = self.x + alpha*p
        fxprev, gxprev = fx, gx
        fx, gx = self.f(self.x)
        y = gx - gxprev
        s = self.x - xprev
        rho = 1.0 / np.dot(y, s)
        if len(self.old_dirs) == self.history_size:
            self.old_dirs.pop(0)
            self.old_stps.pop(0)
            self.old_fxs.pop(0)
        self.old_dirs.append(s)
        self.old_stps.append(y)
        self.old_fxs.append(fx)
        q = gx
        a = []
        for i in range(len(self.old_dirs)-1, -1, -1):
            a.append(rho * np.dot(self.old_dirs[i], q))
            q = q - a[-1] * self.old_stps[i]
        r = self.H.dot(q)
        for i in range(len(self.old_dirs)):
            b = rho * np.dot(self.old_stps[i], r)
            r = r + self.old_dirs[i] * (a[-i-1] - b)
        self.H = np.dot(np.dot(np.eye(len(self.x)) - rho * np.outer(s, y), self.H), np.eye(len(self.x)) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        return fx, gx


class BFGS:
    ...


class Newton():
    def __init__(self, model, loss_fn, gtol=1e-1):
        self.model = model
        self.loss_fn = loss_fn
        self.gtol = gtol
        self.early_stop = False
    
    def step(self, data, labels, epoch):
        loss = self.model.loss(data, labels, self.loss_fn)
        dW1, db1, dW2, db2 = self.model.gradient()
        
        # Compute Hessian
        hess_W1, hess_b1, hess_W2, hess_b2 = self.model.hessian()
        
        # Compute Newton directions
        p_W1 = -np.linalg.inv(hess_W1) @ dW1
        p_b1 = -np.linalg.inv(hess_b1) @ db1
        p_W2 = -np.linalg.inv(hess_W2) @ dW2
        p_b2 = -np.linalg.inv(hess_b2) @ db2
        
        # Line search to determine optimal step size
        p_list = [p_W1, p_b1, p_W2, p_b2]
        alpha = line_search(self.model, data, labels, self.loss_fn, p_list, alpha=1, beta=0.5) 
        
        self.model.W1 += alpha * p_list[0] 
        self.model.b1 += alpha * p_list[1]
        self.model.W2 += alpha * p_list[2]
        self.model.b2 += alpha * p_list[3]

        if np.linalg.norm(np.concatenate((dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()))) < self.gtol:    
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))  
        print('epoch {}, loss: {}'.format(epoch, loss/data.shape[0]))
        return self.model



class GD():
    def __init__(self, model, loss_fn, lr=1e-3, gtol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.gtol = gtol
        self.early_stop = False

    def step(self, data, labels, epoch):
        # loss = self.model.loss(data, labels, self.loss_fn)
        dW1, db1, dW2, db2 = self.model.gradient()
        if np.linalg.norm(np.concatenate((dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()))) < self.gtol:
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))
        else:
            self.model.W1 -= self.lr * dW1
            self.model.b1 -= self.lr * db1
            self.model.W2 -= self.lr * dW2
            self.model.b2 -= self.lr * db2
        return self.model


class SD():
    def __init__(self, model, loss_fn, gtol=1e-5):
        self.model = model
        self.loss_fn = loss_fn
        self.gtol = gtol
        self.early_stop = False
    
    def step(self, data, labels, epoch):
        loss = self.model.loss(data, labels, self.loss_fn)
        dW1, db1, dW2, db2 = self.model.gradient()
        
        # Compute steepest descent directions
        p_list = [-dW1, -db1, -dW2, -db2]
        
        # Line search to determine optimal step size
        alpha = line_search(self.model, data, labels, self.loss_fn, p_list, alpha=1, beta=0.5) 
        
        self.model.W1 += alpha * p_list[0] 
        self.model.b1 += alpha * p_list[1]
        self.model.W2 += alpha * p_list[2]
        self.model.b2 += alpha * p_list[3]

        if np.linalg.norm(np.concatenate((dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()))) < self.gtol:    
            self.early_stop = True
            print('epoch {} is converged'.format(epoch))  
        return self.model