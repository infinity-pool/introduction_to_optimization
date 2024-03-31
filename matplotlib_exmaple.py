from matplotlib import pyplot as plt
import numpy as np

# x, y 값 각 입력
plt.plot([1, 2, 3], [110, 130, 120])
plt.show()

# label, title
plt.plot(['Seoul', 'Paris', 'Seattle'], [30, 25, 55])
plt.xlabel('City')
plt.ylabel('Response')
plt.title('Experiment Result')
plt.show()

# 그래프 2개와 plt.legend([라인1범례, 라인2범례]) 함수를 이용하여 범례 추가
plt.plot([1, 2, 3], [1, 4, 9])
plt.plot([2, 3, 4], [5, 6, 7])
plt.xlabel('Sequence')
plt.ylabel('Time(sec)')
plt.title('Experiment Result')
plt.legend(['Mouse', 'Cat'])
plt.show()

# Bar 차트
y = [5, 3, 5, 2, 19, 35, 3, 1]
x = range(len(y))
plt.bar(x, y, width=0.7, color='red')
plt.show()

# numpy 사용해 그래프 그리기
y_test = np.random.randn(30)
x_test = np.arange(y_test.size)
plt.plot(x_test, y_test)
plt.show()

# 예제. numpy 함수 arange로 x축 값이 -5~5인 2차 함수 그래프 그리기
x_ex = np.arange(-5, 6)
y_ex = np.square(x_ex)
plt.plot(x_ex, y_ex)
plt.show()