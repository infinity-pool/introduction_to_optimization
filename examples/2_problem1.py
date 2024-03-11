import numpy as np
import matplotlib.pyplot as plt

iter_num = 100
datas = np.zeros(shape = (iter_num, ), dtype = np.double)
ans = 0
for i in range(1, iter_num + 1):
    ans = ans + i
    datas[i - 1] = ans

print("Answer : ", ans)
plt.figure(1)
plt.plot(range(0, iter_num), datas)
plt.show()