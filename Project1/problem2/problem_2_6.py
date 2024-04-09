# 2018016244 추현욱
# Problem 2-6
import numpy as np                  # import numpy module
import matplotlib.pyplot as plt     # import matplotlib.pyplot module
import funcs                        # import custom module that defines f, grad_f
from scipy.io import loadmat        # import loadmat from scipy.io

A = np.load('./datas/A.npy') # Load predefined A matrix
Q = np.load('./datas/Q.npy') # Load predefined Q matrix
b = np.load('./datas/b.npy') # Load predefined b vector

x_optimal_2_data = loadmat('./datas/x_optimal_by_2.mat') # Load x^* data obtained by 2
x_optimal_2 = x_optimal_2_data['x_optimal'] # Load x^*
x_optimals_3 = np.squeeze(np.load('./datas/x_optimals_by_3.npy')) # Load x^*_Gradient obtained by 3
x_optimals_4 = np.squeeze(np.load('./datas/x_optimals_by_4.npy')) # Load x^*_SteepestGradient obtained by 3
x_optimals_5 = np.squeeze(np.load('./datas/x_optimals_by_5.npy')) # Load x^*_Nesterov obtained by 3

difference1 = np.squeeze(x_optimal_2) - x_optimals_3 # x* - x*_Gradient
difference2 = np.squeeze(x_optimal_2) - x_optimals_4 # x* - x*_SteepestGradient
difference3 = np.squeeze(x_optimal_2) - x_optimals_5 # x* - x*_Nesterov-2
norms1 = [np.linalg.norm(x_optimal_2 - optimal) for optimal in x_optimals_3] # ∣∣x* - x*_Gradient∣∣
norms2 = [np.linalg.norm(x_optimal_2 - optimal) for optimal in x_optimals_4] # ∣∣x* - x*_SteepestGradient∣∣
norms3 = [np.linalg.norm(x_optimal_2 - optimal) for optimal in x_optimals_5] # ∣∣x* - x*_Nesterov-2∣∣

plt.figure(figsize=(12, 9)) # Set Plot size
# Plot x* - x*_Gradient
plt.subplot(321)
plt.grid(True)
plt.plot(difference1)
plt.xlabel("Iteration") # x label
plt.ylabel("x* - x*_Gradient") # y label

# Plot x* - x*_SteepestGradient
plt.subplot(323)
plt.grid(True)
plt.plot(difference2)
plt.xlabel("Iteration") # x label
plt.ylabel("x* - x*_SteepestGradient") # y label

# Plot x* - x*_Nesterov-2
plt.subplot(325)
plt.grid(True)
plt.plot(difference3)
plt.xlabel("Iteration") # x label
plt.ylabel("x* - x*_Nesterov-2") # y label

# Plot ∣∣x* - x*_Gradient∣∣
plt.subplot(322)
plt.grid(True)
plt.plot(norms1)
plt.xlabel("Iteration") # x label
plt.ylabel("∣∣x* - x*_Gradient∣∣") # y label

# Plot ∣∣x* - x*_SteepestGradient∣∣
plt.subplot(324)
plt.grid(True)
plt.plot(norms2)
plt.xlabel("Iteration") # x label
plt.ylabel("∣∣x* - x*_SteepestGradient∣∣") # y label

# Plot ∣∣x* - x*_Nesterov-2∣∣
plt.subplot(326)
plt.grid(True)
plt.plot(norms3)
plt.xlabel("Iteration") # x label : iteration
plt.ylabel("∣∣x* - x*_Nesterov-2∣∣") # y label : cost (converging to optimal cost)

plt.show() # Show convergence plot