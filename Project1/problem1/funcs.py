# 2018016244 추현욱
import numpy as np                  # import numpy module
import math                         # import math module
from scipy import optimize          # import optimize module

# Define the 1st function to optimize. x = np.array([x_1, x_2])
def f1(x):
    x_1 = x[0]  # x_1
    x_2 = x[1]  # x_2
    return 1 + 2*x_1 + 3*(x_1**2 + x_2**2) + 4*x_1*x_2  # return the cost value

# Define the gradient of 1st function. x = np.array([x_1, x_2])
def grad_f1(x):
    x_1 = x[0]  # x_1
    x_2 = x[1]  # x_2
    return np.array([6*x_1 + 4*x_2 + 2, 4*x_1 + 6*x_2]) # return the gradient vector of x

# Define the 2nd function to optimize. x = np.array([x_1, x_2])
# return the cost value calculated using matrix multiplication
def f2(x):
    return (x.reshape(1, 2) @ np.array([[3, 3], [1, 3]]) @ x + np.array([16, 23]).reshape(1, 2) @ x + (math.pi)**2)[0]

# Define the gradient of 2nd function. x = np.array([x_1, x_2])
def grad_f2(x):
    x_1 = x[0]  # x_1
    x_2 = x[1]  # x_2
    return np.array([6*x_1 + 4*x_2 + 16, 4*x_1 + 6*x_2 + 23]) # return the gradient vector of x

# Define the 3rd function to optimize. x_vec = np.array([x, y])
def f3(x_vec):
    x = x_vec[0]  # x
    y = x_vec[1]  # y
    return 3*(x**2 + y**2) + 4*x*y + 5*x + 6*y + 7  # return the cost value

# Define the gradient of 3rd function. x_vec = np.array([x, y])
def grad_f3(x_vec):
    x = x_vec[0]  # x
    y = x_vec[1]  # y
    return np.array([6*x + 4*y + 5, 4*x + 6*y + 6]) # return the gradient vector of x

# Define function to get optimal step size for steepest gradient descent algorithm
def get_steepest_stepsize(f, grad_f, x_k):
    return optimize.minimize_scalar(lambda alpha: f(x_k-alpha*grad_f(x_k)), method='brent').x