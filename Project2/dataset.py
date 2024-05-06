# Introduction to optimization Project 2
# Group 11 - 2018016244 추현욱

# Define Item class
class Item:
    def __init__(self, num, w, v):
        self.num = num  # number of item
        self.w = w      # weight
        self.v = v      # value

# Knapsack capacity
capacity1 = 100
capacity2 = 30

# item list 1
dataset1 = [
    Item(1, 7, 42),
    Item(2, 11, 88),
    Item(3, 3, 29),
    Item(4, 5, 61),
    Item(5, 3, 16),
    Item(6, 9, 36),
    Item(7, 10, 44),
    Item(8, 6, 63),
    Item(9, 16, 92),
    Item(10, 13, 86),
    Item(11, 12, 49),
    Item(12, 7, 71),
    Item(13, 5, 40),
    Item(14, 7, 82),
    Item(15, 16, 50),
    Item(16, 19, 32),
    Item(17, 14, 96),
    Item(18, 15, 89),
    Item(19, 17, 50),
    Item(20, 15, 33),
    Item(21, 19, 52),
    Item(22, 8, 84),
    Item(23, 12, 75),
    Item(24, 8, 17),
    Item(25, 17, 14),
    Item(26, 17, 22),
    Item(27, 13, 71),
    Item(28, 18, 68),
    Item(29, 4, 55),
    Item(30, 9, 20)
]

# item list 2
dataset2 = [
    Item(1, 15, 80),
    Item(2, 8, 30),
    Item(3, 5, 95),
    Item(4, 19, 20),
    Item(5, 10, 70),
    Item(6, 3, 45),
    Item(7, 2, 60),
    Item(8, 17, 10),
    Item(9, 6, 85),
    Item(10, 12, 50)
]