import numpy as np

# numpy를 이용하여 array 정의하기. python list 활용.
data1 = [1, 2, 3, 4, 5]
arr1 = np.array(data1)
print(arr1)
print(arr1.shape)
print('\n')

# np.zeros()는 인자로 받는 크기만큼, 모든 요소가 0인 array를 만든다.
zeros_arr1 = np.zeros(10)
zeros_arr2 = np.zeros((3, 5))
print(zeros_arr1)
print(zeros_arr2)
print('\n')

# np.ones()는 인자로 받는 크기만큼, 모든 요소가 1인 array를 만든다.
ones_arr1 = np.ones(10)
ones_arr2 = np.ones((2, 10))
print(ones_arr1)
print(ones_arr2)
print('\n')

# array indexing 1D
# np.arrange()는 인자로 받는 값 만큼 1씩 증가하는 1차원 array를 만든다. 이 때, 하나의 인자만 입력하면 0~입력한 인자, 값 만큼의 크기를 가진다.
arange_arr1 = np.arange(10)
print(arange_arr1)
print(arange_arr1[0])
print(arange_arr1[3:9])
print(arange_arr1[:])
print('\n')

# array indexing 2D
arr2 = np.array(([1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]))
print(arr2)
print(arr2[0, 0])
print(arr2[2, :])
print(arr2[:, 3])
print('\n')

# array 연산 - 같은 위치에 있는 요소끼리 연산
arr3 = np.array(([1, 2, 3],
                 [4, 5, 6]))
arr4 = np.array(([10, 11, 12],
                 [13, 14, 15]))
print(arr3.shape, arr4.shape)
arr_sum = arr3 + arr4
arr_sub = arr3 - arr4
arr_mul = arr3 * arr4
arr_div = arr3 / arr4
print(arr_sum)
print(arr_sub)
print(arr_mul)
print(arr_div)
print('\n')

arr5 = arr3 * 10
print(arr5)
print('\n')

# 각 요소에 대해 제곱
arr6 = arr3 ** 2
print(arr6)
print('\n')

# numpy 함수
# 지정한 크기로 랜덤 요소 생성
arr7 = np.random.randn(5, 3)
print(arr7)

# 각 성분의 절대값 계산
print(np.abs(arr7))

# 각 성분의 제곱 계산
print(np.square(arr7))

# 각 성분을 무리수 e의 지수로 넣어 계산
print(np.exp(arr7))
print('\n')

# reshape
test_matrix = np.array([[1, 2, 3, 4],
               [1, 2, 3, 4]])
print(test_matrix.shape)
reshape_matrix = test_matrix.reshape(8, )
print(reshape_matrix)
print(reshape_matrix.shape)
print('\n')

# dot product
a = np.array([3, 4])
b = np.array([4, 5])
dot_prod = np.dot(a, b) # 3*4 + 4*5 = 12 + 20 = 32
print(dot_prod)

# 예제. [[1, 2, 3], [4, 5, 6]] 과 [[7, 8], [9, 10], [11, 12]]를 만들고 내적 계산
mat_a = np.arange(1, 7).reshape(2, 3)
mat_b = np.arange(7, 13).reshape(3, 2)
res_dot_prod = np.dot(mat_a, mat_b)
print(res_dot_prod)
print('\n')