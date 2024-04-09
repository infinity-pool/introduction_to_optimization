# 2018016244 추현욱
# To compare with same matrix, save the matrix file first
# Generate Q, A, b
import numpy as np # import numpy module
import scipy.io    # import scipy.io module

while (True):
    A = np.random.randn(1000, 1000) # Random A matrix(1000x1000)
    # rho = 0.1 # arbitrarily chosen value of rho
    rho = max(np.linalg.eigvals(A.T @ A)) + 0.1
    Q = A @ A.T + rho * np.eye(1000) # Q = AA^T + rho*I
    if (np.all(np.linalg.eigvals(Q)) >= 0): # To make Q positive semi-definite
        break

b = np.random.randn(1000, 1) # Random b vector(1000x1)

np.save('./datas/A.npy', A) # save matrix A file used in numpy(Python)
np.save('./datas/Q.npy', Q) # save matrix Q file used in numpy(Python)
np.save('./datas/b.npy', b) # save vector b file used in numpy(Python)

scipy.io.savemat('./datas/A.mat', {'A': A}) # save matrix A file used in MATLAB
scipy.io.savemat('./datas/Q.mat', {'Q': Q}) # save matrix Q file used in MATLAB
scipy.io.savemat('./datas/b.mat', {'b': b}) # save vector b file used in MATLAB