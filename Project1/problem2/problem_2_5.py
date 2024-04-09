# 2018016244 추현욱
# Problem 2-5
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module
import funcs                        # import custom module that defines f, grad_f

A = np.load('./datas/A.npy') # Load predefined A matrix
Q = np.load('./datas/Q.npy') # Load predefined Q matrix
b = np.load('./datas/b.npy') # Load predefined b vector

MAX_ITER_NUM = 100 # Maximum iteration
STEP_SIZE = 0.0001 # Set step size (t_k)

alpha_datas = np.zeros((MAX_ITER_NUM, ), dtype = np.double) # Store alphas
beta_datas = np.zeros((MAX_ITER_NUM, ), dtype = np.double) # Store betas
x_datas = np.zeros((MAX_ITER_NUM, 1000, 1), dtype = np.double) # Store x vectors
y_datas = np.zeros((MAX_ITER_NUM, 1000, 1), dtype = np.double) # Store y vectors
f_datas = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration

alpha_datas[0] = 1 # Set initial alpha(alpha_1)
x_datas[0] = np.ones(shape = (1000, 1)) # Set initial condition
y_datas[0] = np.zeros(shape = (1000, 1)) # Set initial condition

f_datas[0] = funcs.f(x_datas[0], Q, b)   # Calculate initial function value

for k in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    alpha_datas[k + 1] = funcs.getNextAlpha(alpha_datas[k]) # Calculate alpha_(k+1)
    beta_datas[k] = funcs.getBeta(alpha_datas[k], alpha_datas[k + 1]) # Calculate beta_k
    y_datas[k + 1] = x_datas[k] - STEP_SIZE * funcs.grad_f(x_datas[k], Q, b) # Calculate y_(k+1) using Nesterov-2
    x_datas[k + 1] = y_datas[k + 1] + beta_datas[k] * (y_datas[k + 1] - y_datas[k]) # Calculate x_(k+1) using Nesterov-2
    f_datas[k + 1] = funcs.f(x_datas[k + 1], Q, b) # Calculate f(x_(k+1))

    # Break when ∣f(x_(k+1)) - f(x_k)∣ < 10^-5 and ∣∣x_(k+1) - x_k∣∣ < 10^-5
    if (abs(f_datas[k + 1] - f_datas[k]) < 1e-5 and np.linalg.norm(x_datas[k + 1] - x_datas[k]) < 1e-5):
        break

print("Iteration :", k + 1) # Print iteration count
print("Optimal Cost : ", f_datas[k + 1]) # Print optimal cost
print("Optimal Solution : {}".format(x_datas[k + 1])) # Print optimaizer

plt.plot(range(k + 2), f_datas[:k + 2]) # Plot f that converges to optimal cost
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("Optimal Cost") # y label : cost (converging to optimal cost)
plt.title("Nesterov-2 Algorithm(Problem 2-5)") # title

plt.show() # Show convergence plot

np.save('./datas/x_optimals_by_5.npy', x_datas[:k+2]) # save x_datas

"""
Iteration : 38
Optimal Cost :  -0.09963156088734723
Optimal Solution : [[-1.68113100e-04]
...
 [-6.67406925e-05]]
"""