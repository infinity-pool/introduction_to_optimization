# 2018016244 추현욱
# Problem 5-2
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module

# Define f(x)
def f(x, A, Q, b):
    return ((A @ x - b).T @ Q @ (A @ x - b))[0]

# Define gradient of f(x)
def grad_f(x, A, Q, b):
    return 2 * A.T @ Q @ (A @ x - b)

n = 100 # Given n = 100
m = 50  # Given m = 50

# Generate random A, b, Q
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
Q = np.eye(m) * np.random.uniform(low=1.0, high=2.0, size=(m, m)) # To ensure Q is positive definite

# Optimal Solution via 1 (Analytical method). x^* = (A^T Q A)^-1 A^T Q b
x_analytical_opt = (np.linalg.inv(A.T @ Q @ A)) @ A.T @ Q @ b
f_analytical_opt = f(x_analytical_opt, A, Q, b)[0]

MAX_ITER_NUM = 1000 # Maximum iteration
STEP_SIZE = 0.001 # Set constant step size
INITIAL_CONDITION = np.zeros(shape = (n, 1)) # Set initial condition

x_datas = np.zeros((MAX_ITER_NUM, n, 1), dtype = np.double) # Store x vectors
f_datas = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration
x_datas[0] = INITIAL_CONDITION  # Set initial point
f_datas[0] = f(x_datas[0], A, Q, b)   # Calculate initial function value

for k in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    x_datas[k + 1] = x_datas[k] - STEP_SIZE * grad_f(x_datas[k], A, Q, b) # Calculate x_(k+1) using 'gradient descent algorithm' with constant stepsize
    f_datas[k + 1] = f(x_datas[k + 1], A, Q, b) # Calculate f(x_(k+1))

    # Break when ∣f(x_(k+1)) - f(x_k)∣ < 10^-5 and ∣∣x_(k+1) - x_k∣∣ < 10^-5
    if (abs(f_datas[k + 1] - f_datas[k]) < 1e-5 and np.linalg.norm(x_datas[k + 1] - x_datas[k]) < 1e-5):
        break

print("Iteration :", k + 1) # Print iteration count
print("Optimal Cost (Analytical) : ", f_analytical_opt) # Print analytical optimal cost
print("Optimal Cost (Gradient Descent) : ", f_datas[k + 1]) # Print optimal cost
print("∣∣(Analytical Optimal Solution) - (Gradient Descent Optimal Solution)∣∣ : ", np.linalg.norm(x_analytical_opt - x_datas[k + 1])) # Print ∣∣(Analytical Optimal Solution) - (Gradient Descent Optimal Solution)∣∣

plt.plot(range(k + 2), f_datas[:k + 2]) # Plot f that converges to optimal cost
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("Optimal Cost") # y label : cost (converging to optimal cost)
plt.title("Gradient Descent Algorithm(Problem 5-2, stepsize = {})".format(STEP_SIZE)) # title

plt.show() # Show convergence plot