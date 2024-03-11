import numpy as np
import matplotlib.pyplot as plt

n = 100
a_n = np.zeros(shape = (n, ), dtype = np.double)
a_n[0] = 2**0.5

for i in range(1, n):
    a_n[i] = (2*a_n[i - 1])**0.5

print("Answer : ", a_n[n - 1])
plt.figure(1)
plt.plot(range(0, n), a_n)
plt.show()