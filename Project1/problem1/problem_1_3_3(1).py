# 2018016244 추현욱
# Problem 1-3_func3 _ constant stepsize (a_k = 0.01)
# Convergence speed : (Using const stepsize) < (Using diminishing stepsize)
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module
import funcs                        # import custom module that defines f1, f2, f3

MAX_ITER_NUM = 1000 # Maximum iteration
STEP_SIZE = 0.01 # Set constant step size

x_datas = np.zeros((MAX_ITER_NUM, 2, ), dtype = np.double) # Store x vectors
f_datas = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration
x_datas[0] = np.array([0, 0])       # Set initial point
f_datas[0] = funcs.f3(x_datas[0])   # Calculate initial function value

for k in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    x_datas[k + 1] = x_datas[k] - STEP_SIZE * funcs.grad_f3(x_datas[k]) # Calculate x_(k+1) using 'gradient descent algorithm' with constant stepsize
    f_datas[k + 1] = funcs.f3(x_datas[k + 1]) # Calculate f(x_(k+1))

    # Break when ∣f(x_(k+1)) - f(x_k))∣ < 10^-6 and ∣∣grad(x_k)∣∣ < 10^-6
    if ((abs(f_datas[k + 1] - f_datas[k]) < 1e-6) and np.linalg.norm(funcs.grad_f3(x_datas[k])) < 1e-6):
        break

print("Iteration :", k + 1) # Print iteration count
print("Optimal Solution(f3) : ({}, {})".format(x_datas[k + 1][0], x_datas[k + 1][1])) # Print optimaizer
print("Optimal Cost(f3) : ", f_datas[k + 1]) # Print optimal cost

# Plot 'Optimal Solution' & 'Optimal Cost'
plt.subplot(211)
plt.plot(range(k + 2), x_datas[:k + 2]) # Plot x that converges to optimal solution
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("Optimal Solution") # y label : x (converging to optimal solution)
plt.title("Gradient Descent Algorithm(Problem 1-3, f3, const stepsize)") # title

plt.subplot(212)
plt.plot(range(k + 2), f_datas[:k + 2]) # Plot f3 that converges to optimal cost
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("Optimal Cost") # y label : cost (converging to optimal cost)

plt.show() # Show 2 plots above