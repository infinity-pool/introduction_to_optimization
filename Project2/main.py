# Introduction to optimization Project 2
# Group 11 - 2018016244 추현욱

import dataset    # import custom dataset
import funcs_def  # import custom module that defines "greedy_knapsack", "DP_knapsack", "BB_knapsack" and "LP_knapsack" functions

# Test 4 algorithms on two datasets
items_list = [dataset.dataset1, dataset.dataset2]   # list of items. item(number, weight, value)
capacities_list = [dataset.capacity1, dataset.capacity2]  # list of knapsack's capacities

# Test on two datasets
for i in range(len(items_list)):
    items = items_list[i]   # item list
    c = capacities_list[i]  # knapsack capacity
    z_greedy, x_greedy = funcs_def.greedy_knapsack(items, c)  # optimal cost & x^* obtained by Greedy algorithm
    z_LP, x_LP = funcs_def.LP_knapsack(items, c)  # optimal cost & x^* obtained by Linear Programming
    z_DP, x_DP = funcs_def.DP_knapsack(items, c)  # optimal cost & x^* obtained by Dynamic Programming
    z_BB, x_BB = funcs_def.BB_knapsack(items, c)  # optimal cost & x^* obtained by Branch and Bound algorithm

    # print the result(z, x^*)
    print("<<Dataset {} - num of items : {}, capacity : {}>>".format(i + 1, len(items), c))  # dataset info
    funcs_def.printResult(z_greedy, x_greedy, "greedy")  # print the result of Greedy algorithm
    funcs_def.printResult(z_LP, x_LP, "LP")  # print the result of relaxation Linear Programming
    funcs_def.printResult(z_DP, x_DP, "DP")  # print the result of Dynamic Programming
    funcs_def.printResult(z_BB, x_BB, "BB")  # print the result of Branch and Bound algorithm
    print()