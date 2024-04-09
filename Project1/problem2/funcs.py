# 2018016244 추현욱
import numpy as np                  # import numpy module
from scipy import optimize          # import optimize module

# Define cost function
def f(x, Q, b):
    return (0.5 * x.T @ Q @ x - b.T @ x)[0]

# Define gradient of cost function
def grad_f(x, Q, b):
    return Q @ x - b

# For Steepest Gradient algorithm
# Define function to get optimal step size
def get_steepest_stepsize(f, grad_f, x_k, Q, b):
    return optimize.minimize_scalar(lambda alpha: f((x_k-alpha*grad_f(x_k, Q, b)), Q, b), method='brent').x

# For Nesterov-2 algorithm
# Calculate alpha_(k+1) from alpha_k
def getNextAlpha(alpha_k):
    return (np.sqrt(alpha_k**4 + 4*(alpha_k**2)) - alpha_k**2) / 2

# For Nesterov-2 algorithm
# Calculate beta_k from alpha_k and alpha_(k+1)
def getBeta(alpha_k, alpha_k_plus1):
    return (alpha_k * (1 - alpha_k)) / (alpha_k**2 + alpha_k_plus1)