import numpy as np
import matplotlib.pyplot as plt

iter_num = 100
datas = np.zeros(shape = (iter_num, ), dtype = np.double)
ans = 0

# range starts from 1 since prevent division by zero.
for i in range(1, iter_num + 1):
    datas[i - 1] = ((5/i)+(3/(i**2)))/((1/i)-(2/(i**3)))
    ans = datas[i - 1]

print("Answer : ", ans)
plt.figure(1)
plt.plot(range(0, iter_num), datas)
plt.show()