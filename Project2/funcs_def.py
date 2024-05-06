# Introduction to optimization Project 2
# Group 11 - 2018016244 추현욱

import dataset  # import custom dataset
from scipy.optimize import linprog  # import linprog module
from heapq import heappush, heappop # import heapq since branch and bound using priority queue

def LP_knapsack(items, c):  # input : items, capacity
    z = 0   # profit of algorithm
    x = []  # list of x_i

    reversed_values = [-item.v for item in items] # linprog minimize the cost function. Values should be negative.
    A = [[item.w for item in items]] # weight of each items
    b = [c] # capacity of knapsack
    x_bounds = [(0, 1) for _ in range(len(items))] # x boundary

    result = linprog(reversed_values, A_ub=A, b_ub=b, bounds=x_bounds, method='highs') # Run linprog function

    z = -result.fun # optimal profit by LP
    x = result.x # optimal knapsack strategy (How much item is contained. between 0 and 1 is possible)

    return z, x # return optimal profit & optimal strategy


def greedy_knapsack(items, c):  # input : items, capacity
    z = 0   # profit of algorithm
    x = []  # list of item numbers in optimal strategy set
    weight = 0 # weight of algorithm

    sorted_items = sorted(items, key=lambda x: x.v/x.w, reverse=True) # Sorting items by "value/weight" in decreasing order
    for item in sorted_items:   # Start to find solution
        if (weight + item.w <= c):  # check whether we can put in this item in the knapsack. if it's possible, jump in.
            z += item.v # add item's value to total profit
            x.append(item.num)  # add item's num to x
            weight += item.w    # add item's weight to total weight

    return z, x # return optimal profit & optimal strategy


def DP_knapsack(items, c):  # input : items, capacity
    n = len(items)  # number of items
    z = 0   # profit of algorithm
    x = []  # list of item numbers in solution set
    dp_table = [[0 for _ in range(n + 1)] for _ in range(c + 1)]  # z_j(d) table

    # for loop starts from j = 1, d = 1 since elements of first row and first column are all zero
    for j in range(1, n + 1):
        for d in range(1, c + 1):
            if (d < items[j - 1].w):    # if item j can't be put in the knapsack
                dp_table[d][j] = dp_table[d][j - 1]  # z_j(d) = z_j-1(d)
            else:   # if item j can be put in the knapsack
                dp_table[d][j] = max(dp_table[d][j - 1], dp_table[d - items[j - 1].w][j - 1] + items[j - 1].v)  # z_j(d) = max(z_j-1(d), z_j-1(d - w_j) + p_j)

    z = dp_table[c][n]  # dp_table[c][n] is the optimal solution

    # the code below is that backtracks dp_table to find x(optimal strategy)
    curr_capacity = c   # set current capacity as c
    for j in range(n, 0, -1):  # start backtrack
        if dp_table[curr_capacity][j] != dp_table[curr_capacity][j - 1]:  # if item j is in the optimal strategy set, dp_table[d][j-1] and dp_table[d][j] are different
            x.append(items[j - 1].num)      # append item j in x (!! item j has j - 1 index in the items list !!)
            curr_capacity -= items[j - 1].w # change dp_table row (d)
    x.reverse()  # sort x as increasing order to easily compare with other algorithms

    return z, x  # return optimal profit & optimal strategy


# define state Node class
class Node:
    def __init__(self, level=None, value=None, weight=None, bound=None, contains=[]):
        self.level = level          # current node's level (index)
        self.value = value          # sum of item values
        self.weight = weight        # sum of item weights
        self.bound = bound          # current node's bound
        self.contains = contains    # contained item numbers in current node
    
    def __lt__(self, other):        # in priority queue, higher bound node has higher priority
        return self.bound > other.bound

# define the function that calculate current node's upper bound
def bound(node, c, items):
    n = len(items)  # number of items
    if node.weight >= c: # not feasible case
        return 0         # return bound as 0
    else:   # feasible case
        bound = node.value  # set initial bound as node.value
        j = node.level + 1  # candiate item's index to include or exclude
        totweight = node.weight  # set initial totweight as node.weight
        while j < n and totweight + items[j].w <= c:  # calculate totweight and bound
            totweight += items[j].w  # calculate totweight
            bound += items[j].v  # calculate bound
            j += 1
        if j < n:
            bound += (c - totweight) * (items[j].v / items[j].w)  # calculate bound
        return bound

# Best first search with branch and bound prunning!!
def BB_knapsack(items, c):  # input : items, capacity
    n = len(items)  # number of items
    z = 0   # profit of algorithm
    x = []  # list of item numbers in solution set

    sorted_items = sorted(items, key=lambda x: x.v/x.w, reverse=True) # Sorting items by "value/weight" in decreasing order
    priorityQ = []  # Using priority queue
    curr_node = Node(level=-1, value=0, weight=0, contains=[])  # set initial current node as root of state tree
    heappush(priorityQ, (-curr_node.bound if curr_node.bound else 0, curr_node))
    while priorityQ:    # iterate while priority queue is not empty
        curr_node = heappop(priorityQ)[1]   # pop current node from priority queue(heap)
        if curr_node.level == n - 1:    # if current node is last node, stop branching
            continue
        j = curr_node.level + 1  # next item's index to decide include or exclude
        # new node including item j
        node_include_j = Node(level=j, weight=curr_node.weight+sorted_items[j].w, value=curr_node.value+sorted_items[j].v, contains=curr_node.contains+[sorted_items[j].num])
        node_include_j.bound = bound(node_include_j, c, sorted_items)  # calculate new node's bound
        if node_include_j.weight <= c and node_include_j.value > z:  # case when max_profit is changed
            z = node_include_j.value     # update max_profit(optimal solution, z)
            x = node_include_j.contains  # update optimal strategy, x
        if node_include_j.bound > z and node_include_j.weight <= c:  # check whether new node is promising
            heappush(priorityQ, (-node_include_j.bound, node_include_j))  # if new node is promising, push to priority queue
        # new node excluding item j
        node_exclude_j = Node(level=j, weight=curr_node.weight, value=curr_node.value, contains=curr_node.contains)
        node_exclude_j.bound = bound(node_exclude_j, c, sorted_items)  # calculate new node's bound
        if node_exclude_j.bound > z and node_exclude_j.weight <= c:  # check whether new node is promising
            heappush(priorityQ, (-node_exclude_j.bound, node_exclude_j))  # if new node is promising, push to priority queue

    return z, x  # return optimal profit & optimal strategy



# This function receives z, x and algorithm name as input and prints the result of solution of knapsack problem by algorithms
# No need to explain for each lines
def printResult(z, x, used_algorithm):
    print("[Result of '{}']".format(used_algorithm))
    print(" - z(optimal cost) :", z)
    print(" - x(Optimal Knapsack strategy) : ", end='')
    if (used_algorithm == "LP"):
        for i in enumerate(x):
            if (i[1] != 0):
                print("[{}]({}) ".format(i[0] + 1, i[1]), end='')
    else:
        for i in x:
            print("[{}] ".format(i), end='')
        if (used_algorithm in ("greedy", "BB")):
            x.sort()
            print()
            print(" - Sorted x(Optimal Knapsack strategy) : ", end='')
            for i in x:
                print("[{}] ".format(i), end='')
    print()