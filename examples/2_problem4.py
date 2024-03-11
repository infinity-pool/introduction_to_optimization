import numpy as np
import matplotlib.pyplot as plt

# Similar answer from iter_num = 100 to iter_num = 10000
n = 100
datas = np.zeros(shape = (n, ), dtype = np.double)
ans = 0

for i in range(0, n):
    k = i + 1
    ans = ans + ((k**2) + 2*k*n) / ((k**3) + 3*(k**2)*n + (n**3))
    datas[i] = ans

# datas[n-1] == ans
print("Answer : ", datas[n - 1])
plt.figure(1)
plt.plot(range(0, n), datas)
plt.show()