# 2018016244 추현욱
# Problem 2-3
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module
import funcs                        # import custom module that defines f, grad_f

A = np.load('./datas/A.npy') # Load predefined A matrix
Q = np.load('./datas/Q.npy') # Load predefined Q matrix
b = np.load('./datas/b.npy') # Load predefined b vector

MAX_ITER_NUM = 100 # Maximum iteration
STEP_SIZE = 0.0002 # Set constant step size
INITIAL_CONDITION = np.ones(shape = (1000, 1)) # Set initial condition

x_datas = np.zeros((MAX_ITER_NUM, 1000, 1), dtype = np.double) # Store x vectors
f_datas = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration
x_datas[0] = INITIAL_CONDITION  # Set initial point
f_datas[0] = funcs.f(x_datas[0], Q, b)   # Calculate initial function value

for k in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    x_datas[k + 1] = x_datas[k] - STEP_SIZE * funcs.grad_f(x_datas[k], Q, b) # Calculate x_(k+1) using 'gradient descent algorithm' with constant stepsize
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
plt.title("Gradient Descent Algorithm(Problem 2-3, stepsize = {})".format(STEP_SIZE)) # title

plt.show() # Show convergence plot

np.save('./datas/x_optimals_by_3.npy', x_datas[:k+2]) # save x_datas

# Result is same as result of problem 2-2)
"""
Iteration : 25
Optimal Cost :  -0.09963205814984408
Optimal Solution : [[-1.68384532e-04]
...
 [-6.73313036e-05]]
"""