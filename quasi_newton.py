#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:02:57 2023

@author: liut0c
"""
import numpy as np
from numpy import linalg as LA
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

def rosenbrock(x):
    return 10 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

rosen_grad = jax.grad(rosenbrock)
rosen_hessian = jax.hessian(rosenbrock)

def feasible(x_k):
    return True

def backtrack_line_search(f, grad_k, x_k, p_k, t=1, alpha=0.1, beta=0.8):
    while not feasible(x_k + t * p_k):
        t *= beta
        
    while f(x_k + t * p_k) > f(x_k) + alpha * t * grad_k @ p_k: 
        t *= beta
    
    return x_k + t * p_k

# The following function was written by an AI language model created by OpenAI, known as ChatGPT.
def update_inverse_hessian(H, s, y):
    """
    Update the inverse Hessian approximation using the BFGS formula.
    
    Parameters
    ----------
    H : numpy.ndarray
        The current inverse Hessian approximation (n x n matrix).
    s : numpy.ndarray
        The difference between the current and previous parameter vectors (1D array of length n).
    y : numpy.ndarray
        The difference between the current and previous gradient vectors (1D array of length n).
    
    Returns
    -------
    numpy.ndarray
        The updated inverse Hessian approximation (n x n matrix).
    """
    rho = 1 / np.dot(y, s)
    A = np.eye(len(s)) - rho * np.outer(s, y)
    B = np.eye(len(s)) - rho * np.outer(y, s)
    H = A @ H @ B + rho * np.outer(s, s)
    
    return H

def bfgs_bt(f, grad, x_0, max_iteration=100, tol=1e-12): 
    x_history = x_0
    x_k = x_0
    n = len(x_0)
    B_k_inv = np.identity(n)
    error = LA.norm(grad(x_k))
    
    i = 0
    while i < max_iteration and error > tol:
      p_k = - B_k_inv @ grad(x_k)
      x_k_old = x_k
      x_k = backtrack_line_search(f, grad(x_k), x_k, p_k)
      
      s_k = x_k - x_k_old
      y_k = grad(x_k) - grad(x_k_old)
      B_k_inv = update_inverse_hessian(B_k_inv, s_k, y_k)
      x_history = np.vstack((x_history, x_k))
      error = LA.norm(grad(x_k))
      i += 1
    
    return x_history
    
def lbfgs_bt(f, grad, x_0, m=5, max_iteration=100, tol=1e-12):
    def find_direction(g_k, s_k, y_k, k):
        q = g_k 
        alpha = np.zeros(m)
        if k+1 <= m:
            indices = list(range(m-1, m-k-1, -1))
        else:
            indices = list(reversed(range(m)))
        # for i in range(k-1, max(-1, k-m-1), -1):
        for i in indices:
            alpha[i] = np.dot(s_k[i], q) / np.dot(y_k[i], s_k[i])
            print(y_k[i], s_k[i])
            q -= alpha[i] * y_k[i]
        r = q
        for i in reversed(indices):
            beta = np.dot(y_k[i], r) / np.dot(y_k[i], s_k[i])
            r += (alpha[i] - beta) * s_k[i]
        return -r
    
    n = len(x_0)
    x_history = x_0
    x_k = x_0  
    s_k = np.zeros((m, n))
    y_k = np.zeros((m, n))
    error = LA.norm(grad(x_k))
    
    k = 0
    while k < max_iteration and error > tol:   
        g_k = grad(x_k)
        p_k = find_direction(g_k, s_k, y_k, k)
        x_k_old = x_k
        x_k = backtrack_line_search(f, grad(x_k), x_k, p_k)
        
        s_k[:-1] = s_k[1:]
        s_k[-1] = x_k - x_k_old
        y_k[:-1] = y_k[1:]
        y_k[-1] = grad(x_k) - grad(x_k_old)
        x_history = np.vstack((x_history, x_k))
        error = LA.norm(grad(x_k))
        k += 1
    
    return x_history

def newton_bt(f, grad, hess, x_0, max_iteration=100, tol=1e-12): 
    x_history = x_0
    x_k = x_0
    error = LA.norm(grad(x_k))
    
    i = 0
    while i < max_iteration and error > tol:
      p_k = - LA.solve(hess(x_k), grad(x_k))
      x_k = backtrack_line_search(f, grad(x_k), x_k, p_k)

      x_history = np.vstack((x_history, x_k))
      error = LA.norm(grad(x_k))
      i += 1
    return x_history

n = 240
x_1 = np.linspace(-1.2, 1.2, n)
x_2 = np.linspace(-0.5, 1.5, n)

X_1, X_2 = np.meshgrid(x_1, x_2)
Z = rosenbrock([X_1, X_2])
levels_1 = np.linspace(0, 4, 8)
levels_2 = np.linspace(5, 14, 6)
levels = np.concatenate([levels_1, levels_2])

plt.figure(figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
f_contour = plt.contour(X_1, X_2, Z, levels, cmap='jet')
plt.clabel(f_contour, f_contour.levels, fontsize=9) 
plt.plot(1, 1, marker='o')
plt.text(1.03, 1.03, r'$x^*$')
plt.axis('scaled')
plt.title('Contour lines of the Rosenbrock function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

x_opt = np.array([1., 1.])
x_0 = np.array([-1.2, 1.0])
x_trace_newton = newton_bt(rosenbrock, rosen_grad, rosen_hessian, x_0)
x_trace_bfgs = bfgs_bt(rosenbrock, rosen_grad, x_0)
x_trace_lbfgs = lbfgs_bt(rosenbrock, rosen_grad, x_0)
error_trace_newton = [np.abs(rosenbrock(x_i) - rosenbrock(x_opt)) for x_i in x_trace_newton]
error_trace_bfgs = [np.abs(rosenbrock(x_i) - rosenbrock(x_opt)) for x_i in x_trace_bfgs]
error_trace_lbfgs = [np.abs(rosenbrock(x_i) - rosenbrock(x_opt)) for x_i in x_trace_lbfgs]

x_trace = x_trace_newton
plt.plot(x_trace[:, 0], x_trace[:, 1], marker='o', markersize=3, color='magenta', label="Newton's method") 
n_iteration = len(x_trace)
# for x_k, i in zip(x_trace, range(n_iteration - 1)):
#     plt.annotate("", xy=x_k, xycoords='data', xytext=x_trace[i+1,:], textcoords='data', 
#                   arrowprops=dict(arrowstyle="<-",
#                             shrinkA=4, shrinkB=4,
#                             patchA=None, patchB=None,
#                             connectionstyle='arc3,rad=0.3'))     

x_trace = x_trace_bfgs
plt.plot(x_trace[:, 0], x_trace[:, 1], marker='o', markersize=3, color='brown', label="BFGS") 
n_iteration = len(x_trace)
# for x_k, i in zip(x_trace, range(n_iteration - 1)):
#     plt.annotate("", xy=x_k, xycoords='data', xytext=x_trace[i+1,:], textcoords='data', 
#                   arrowprops=dict(arrowstyle="<-",
#                             shrinkA=4, shrinkB=4,
#                             patchA=None, patchB=None,
#                             connectionstyle='arc3,rad=0.3'))
x_trace = x_trace_lbfgs
plt.plot(x_trace[:, 0], x_trace[:, 1], marker='o', markersize=3, color='red', label="L-BFGS") 
n_iteration = len(x_trace)   
plt.legend()

plt.figure(dpi=300, facecolor="w", edgecolor="k")
plt.plot(error_trace_newton, color='magenta', marker='o', label="Newton's method")
plt.plot(error_trace_bfgs, color='brown', marker='.', label='BFGS')
plt.plot(error_trace_lbfgs, color='red', marker='.', label='L-BFGS')
plt.xlim(0)
plt.yscale("log")
plt.xlabel(r"No. of iterations ($k$)")
plt.ylabel(r"$|f^k - f^*|$")
plt.legend()


