# 2018016244 추현욱
# Problem 1-5_func1 _ x^*_Gradient obtained by constant stepsize (a_k = 0.01)
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module
import funcs                        # import custom module that defines f1, f2, f3

MAX_ITER_NUM = 1000 # Maximum iteration
STEP_SIZE = 0.01 # Set constant step size
x_star = np.array([-0.6, 0.4]) # x^* obtained by hand

x_datas_grad = np.zeros((MAX_ITER_NUM, 2, ), dtype = np.double)     # Store x vectors obtained by 'gradient descent method with constant step size'
f_datas_grad = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration obtained by 'gradient descent method with constant step size'
x_datas_steepestgrad = np.zeros((MAX_ITER_NUM, 2, ), dtype = np.double)     # Store x vectors obtained by 'steepest gradient descent method'
f_datas_steepestgrad = np.zeros(shape = (MAX_ITER_NUM, ), dtype = np.double) # Store objective funcion values for each iteration obtained by 'steepest gradient descent method'

norms1 = np.zeros((MAX_ITER_NUM, 2, ), dtype = np.double)   # Store ∣∣x^* - x^*_Gradient∣∣
norms2= np.zeros((MAX_ITER_NUM, 2, ), dtype = np.double)   # Store ∣∣x^* - x^*_SteepestGradient∣∣

x_datas_grad[0] = np.array([0, 0])       # Set initial point
f_datas_grad[0] = funcs.f1(x_datas_grad[0])   # Calculate initial function value
x_datas_steepestgrad[0] = np.array([0, 0])       # Set initial point
f_datas_steepestgrad[0] = funcs.f1(x_datas_steepestgrad[0])   # Calculate initial function value
norms1[0] = np.linalg.norm(x_star - x_datas_grad[0])    # Calculatate initial ∣∣x^* - x^*_Gradient∣∣
norms2[0] = np.linalg.norm(x_star - x_datas_steepestgrad[0])    # Calculatate initial ∣∣x^* - x^*_SteepestGradient∣∣

for k1 in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    x_datas_grad[k1 + 1] = x_datas_grad[k1] - STEP_SIZE * funcs.grad_f1(x_datas_grad[k1]) # Calculate x_(k+1) using 'gradient descent algorithm' with constant stepsize
    f_datas_grad[k1 + 1] = funcs.f1(x_datas_grad[k1 + 1]) # Calculate f(x_(k+1))
    norms1[k1 + 1] = np.linalg.norm(x_star - x_datas_grad[k1 + 1]) # Calculatate ∣∣x^* - x^*_SteepestGradient∣∣

    # Break when ∣f(x_(k+1)) - f(x_k))∣ < 10^-6 and ∣∣grad(x_k)∣∣ < 10^-6
    if ((abs(f_datas_grad[k1 + 1] - f_datas_grad[k1]) < 1e-6) and np.linalg.norm(funcs.grad_f1(x_datas_grad[k1])) < 1e-6):
        break

for k2 in range(0, MAX_ITER_NUM - 1):    # Get x_(k+1) from x_k
    step_size = funcs.get_steepest_stepsize(funcs.f1, funcs.grad_f1, x_datas_steepestgrad[k2]) # Set step size as 'steepest gradient descent'
    x_datas_steepestgrad[k2 + 1] = x_datas_steepestgrad[k2] - step_size * funcs.grad_f1(x_datas_steepestgrad[k2]) # Calculate x_(k+1) using 'steepest gradient descent algorithm'
    f_datas_steepestgrad[k2 + 1] = funcs.f1(x_datas_steepestgrad[k2 + 1]) # Calculate f(x_(k+1))
    norms2[k2 + 1] = np.linalg.norm(x_star - x_datas_steepestgrad[k2 + 1]) # Calculatate ∣∣x^* - x^*_SteepestGradient∣∣

    # Break when ∣f(x_(k+1)) - f(x_k))∣ < 10^-6 and ∣∣grad(x_k)∣∣ < 10^-6
    if ((abs(f_datas_steepestgrad[k2 + 1] - f_datas_steepestgrad[k2]) < 1e-6) and np.linalg.norm(funcs.grad_f1(x_datas_steepestgrad[k2])) < 1e-6):
        break

print('Iteration(Gradient descent algorithm) : ', k1 + 1)
print('Iteration(Steepest gradient descent algorithm) : ', k2 + 1)

# Plot ∣∣x^* - x^*_Gradient∣∣
plt.subplot(211)
plt.plot(range(k1 + 2), norms1[:k1 + 2]) # Plot ∣∣x^* - x^*_Gradient∣∣ that converges to zero
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("∣∣x^* - x^*_Gradient∣∣") # y label : ∣∣x^* - x^*_Gradient∣∣ that converges to zero
plt.title("Gradient Descent Algorithm(Problem 1-5, f1, const stepsize)") # title

# Plot ∣∣x^* - x^*_Gradient∣∣
plt.subplot(212)
plt.plot(range(k2 + 2), norms2[:k2 + 2]) # Plot ∣∣x^* - x^*_SteepestGradient∣∣ that converges to zero
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("∣∣x^* - x^*_SteepestGradient∣∣") # y label : ∣∣x^* - x^*_SteepestGradient∣∣ that converges to zero

plt.show() # Show 2 plots above