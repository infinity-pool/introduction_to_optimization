# 2018016244 추현욱
# Problem 3-1
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module

# Define the function to optimize. x = np.array([x_1, x_2])
def f(x):
    x_1 = x[0]
    x_2 = x[1]
    return 100*((x_2-(x_1)**2)**2) + (1-x_1)**2

# Define the gradient of function. x = np.array([x_1, x_2])
def grad_f(x):
    x_1 = x[0]
    x_2 = x[1]
    return np.array([400*(x_1**3) - 400*x_1*x_2 + 2*x_1 - 2, -200*(x_1**2) + 200*x_2])

MAX_ITER_NUM = 10000 # Maximum iteration
STEP_SIZE = 0.002 # Set constant step size
INITIAL_CONDITION = np.array([0, 0]) # Set initial condition

x_datas = np.zeros((MAX_ITER_NUM, 2, ), dtype = np.double) # Store x vectors
f_datas = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration
x_datas[0] = INITIAL_CONDITION  # Set initial point
f_datas[0] = f(x_datas[0])   # Calculate initial function value

for k in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    x_datas[k + 1] = x_datas[k] - STEP_SIZE * grad_f(x_datas[k]) # Calculate x_(k+1) using 'gradient descent algorithm' with constant stepsize
    f_datas[k + 1] = f(x_datas[k + 1]) # Calculate f(x_(k+1))

    # Break when ∣f(x_(k+1)) - f(x_k)∣ < 10^-5 and ∣∣x_(k+1) - x_k∣∣ < 10^-5
    if (abs(f_datas[k + 1] - f_datas[k]) < 1e-5 and np.linalg.norm(x_datas[k + 1] - x_datas[k]) < 1e-5):
        break

print("Iteration :", k + 1) # Print iteration count
print("Optimal Solution : ({}, {})".format(x_datas[k + 1][0], x_datas[k + 1][1])) # Print optimaizer
print("Optimal Cost : ", f_datas[k + 1]) # Print optimal cost

# Plot 'Optimal Solution' & 'Optimal Cost'
plt.subplot(211)
plt.plot(range(k + 2), x_datas[:k + 2]) # Plot x that converges to optimal solution
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("Optimal Solution") # y label : x (converging to optimal solution)
plt.title("Gradient Descent Algorithm(Problem 3, stepsize = {})".format(STEP_SIZE)) # title

plt.subplot(212)
plt.plot(range(k + 2), f_datas[:k + 2]) # Plot f1 that converges to optimal cost
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("Optimal Cost") # y label : cost (converging to optimal cost)

plt.show() # Show 2 plots above