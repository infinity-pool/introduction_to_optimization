import numpy as np
import matplotlib.pyplot as plt

n = 8
a_n = np.zeros(shape = (n, ), dtype = np.double)
a_n[0] = 1

for i in range(0, n - 1):
    if a_n[i] < 7:
        a_n[i + 1] = 2 * a_n[i]
    else:
        a_n[i + 1] = a_n[i] - 7

ans = 0
for i in range(0, n):
    ans = ans + a_n[i]

print("Answer : ", ans)
plt.figure(1)
plt.scatter(list(range(1, n + 1)), a_n)
plt.show()