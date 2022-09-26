"""
author: Yi Zhen Nicholas Wong
ID: 32577869
Date: 5/5/22
"""

def best_revenue(revenue, travel_days,start):
    """ Returns the maximum revenue the salesman could earn when they start at city start

    :Input: 
    revenue: a list of lists where the interior lists are of length n, len(revenue) = d
    travel_days: square matrix of length n by n
    start: the city the salesman start at

    :Output:
    maximumRevenue - an integer representing the maximum revenue the salesman could get when starting from the city start

    :Time complexity: O(n^2d + n^3) - where n is the length of travel_days and d is the length of revenue
    :Aux space complexity: O(n*d + n^2) - where n is the length of travel_days and d is the length of revenue
    """
    # Edge case when there is no city, means no revenue to earn OR when the start is None due to no city
    if start is None or len(travel_days[0]) == 0:
        return 0
    travel_days = preProcessingForFloyd(travel_days) # O(N^2) time
    travel_days = floydWarshall(travel_days) # O(N^3) time
    memo = createMemo(revenue, travel_days) # O(Nd) time 
    maximumRevenue = maxRevenue(revenue, travel_days, start, memo) # O(n^2d) time 
    return maximumRevenue

def createMemo(revenue, travel_days):
    """ Creating the memoization matrix to solve this problem

    :Input: 
    revenue: a list of lists where the interior lists are of length n, len(revenue) = d
    travel_days: square matrix of length n by n

    :Output:
    memo: a memoization matrix with the same dimensions of revenue

    :Time complexity: O(n*d) where n is the length of the travel_days and d is the len(revenue)
    :Aux space complexity: O(n*d) where n is the length of the travel_days and d is the len(revenue)
    """
    memo = [[None]] * (len(revenue))# O(d)
    for i in range(len(memo)):
        memo[i] = []
        for _ in range(len(travel_days)):
            memo[i].append(-1)
    return memo

def maxRevenue(revenue, travel_days, start, memo):
    """ Returns the maximum revenue when the salesman first starts at the city 'start' to the end of the day travelling to other
    cities and selling the product 

    :Input: 
    revenue: a list of lists where the interior lists are of length n, len(revenue) = d
    travel_days: square matrix of length n by n where indirect paths are also included to the other indexes
    start: the city the salesman start at
    memo: the memoization matrix, d * n dimension

    :Output:
    memo[0][start] - an integer representing the maximum revenue the salesman could get when starting from the city start

    :Time complexity: O(n^2d) - where n is the length of travel_days and d is the length of revenue
    :Aux space complexity: O(1)
    """
    # base case when during the last day, you can only sell at the city
    for i in range(len(travel_days)): # O(n)
        memo[-1][i] = revenue[-1][i]

    days = len(revenue) 
    for day in range(days-2, -1, -1): # O(d)
        for city in range(len(travel_days)): # O(n)
            sell = revenue[day][city] + memo[day+1][city]
            maximumRevenue = sell
            for citi in range(len(travel_days)): # O(n)
                # some cities may not be travellable and is still float inf
                if citi != city and (travel_days[city][citi] + day) < days and travel_days[city][citi] != float("inf"):
                    maximumRevenue = max(maximumRevenue, memo[travel_days[city][citi] + day][citi])
            memo[day][city] = maximumRevenue
    return memo[0][start]

def floydWarshall(matrix):
    """ The floyd Warshall algorithm which allows indirect paths to the index to be included as well

    :Input: 
    matrix: a square matrix

    :Output:
    result_matrix: a sqaure matrix where indirect paths are also included to the other indexes

    :Time complexity: O(n^3) where n is the length of the matrix
    :Aux space complexity: O(1) 

    """
    for k in range(len(matrix)):
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
    return matrix

def preProcessingForFloyd(matrix):
    """ This is the pre-processing to allow floyd-warshall algorithm to work correctly by turning it to inf values instead of 
    -1 for the indirect paths

    :Input: 
    matrix: a square matrix

    :Output:
    result_matrix: a square matrix where instead of the indirect paths are a -1 value, there are of inf value now 

    :Time complexity: O(n^2) where n is the length of the matrix
    :Aux space complexity: O(n^2) where n is the length of the matrix
    """
    result_matrix = []
    for _ in range(len(matrix)):
        result_matrix.append([])

    for i in range(len(matrix)):
        for j in range(len(matrix)): 
            if matrix[i][j] != -1:
                result_matrix[i].append(matrix[i][j])
            else:
                result_matrix[i].append(float("inf"))
    return result_matrix

def hero(attacks):
    """ The lists of multiverses which Master X can defeat the most clones

    :Input: 
    attacks: a non empty list of lists of length N 

    :Output:
    result: the resulting lists we chose from the attacks array in which Master X can defeat the most number of clones

    :Time complexity: O(NlogN) where N is the length of the attacks
    :Aux space complexity: O(N) where N is the length of the attacks
    """
    attacks = merge_sort(attacks, 2) # O(NlogN) time, O(N) space
    memo = [0] * len(attacks)# O(N) space
    breadcrumbs= [-1] * len(attacks) # O(N) space

    # base case
    memo[0] = attacks[0][3] # if there is only the first list, then Master X will certainly attack it. 
    breadcrumbs[0] = -1 # leads to the end

    for index in range(1, len(attacks)): # O(N)
        searching = binary_search(attacks, attacks[index][1]-1) # O(logN)
        include = attacks[index][3]
        if searching != -1:
            include += memo[searching] 
        exclude = memo[index-1]
        if include >= exclude:
            memo[index] = include
            breadcrumbs[index] = searching
        else:
            memo[index] = exclude
            breadcrumbs[index] = index-1
            
    result = backtracking(memo, attacks, breadcrumbs) # O(N) time and space
    return result

def backtracking(memo, attacks, breadcrumbs):
    """ Backtracking the memo and breadcrumbs to find the lists the algorithm picked 

    :Input: 
    memo: the memoization array which consists of maximum clones for each list from 0..N-1 where N is the length of the array
    attacks: non empty list of lists with length of N  
    breadcrumbs: a list of N elements

    :Output:
    lst: the resulting lists we chose from the attacks array based on the memo array

    :Time complexity: O(N) worst case where N is the length of memo, when each list is also chosen by the algorithm
    :Aux space complexity: O(N) worst case where N is the length of memo, when each list is also chosen by the algorithm
    """
    lst = []
    index = len(attacks)-1
    while breadcrumbs[index] != -1: # O(N)
        temp = breadcrumbs[index]
        result = memo[index]
        if memo[temp] != result:
            lst.append(attacks[index])
        index = breadcrumbs[index]
    lst.append(attacks[index])
    return lst

# Got the code from https://stackoverflow.com/questions/29196755/binary-search-for-the-closest-value-less-than-or-equal-to-the-search-value
def binary_search(array, searchValue, startIndex=0):
    """ Binary search to find the index of the closest number in the array that is lesser than or equal to the searchValue! Runs in O(log N) time.

    :Input: 
    array: a list of list where we require to find the index of the list which has the number that is lesser than or equal to the searchValue 
    searchValue: an integer which we need to find in the array, be it lesser is fine as well
    startIndex: an index to start in the array, default is 0

    :Output:
    left - 1: an integer representing an index which is either the exact value of the searchValue in the list or lesser. 

    :Time complexity: O(log n) where n is the length of the array
    :Aux space complexity: O(1)
    """
    left = startIndex
    right = len(array)

    while left < right:
        mid = (left + right) // 2

        if array[mid][2] < searchValue + 1:
            left = mid + 1
        else:
            right = mid

    return left - 1

# got the code from FIT 1045
def merge(lst1, lst2, index): # Helper function for merge sort
    """ Merging 2 lists together 

    :Input: 
    lst1: a sublist containing positive integers
    lst2: a sublist containing positive integers
    index: a positive integer to specify which index to sort on the lists inside the list

    :Output:
    lst: a list of lists which is sorted according to the index of each individual list 

    :Time complexity: O(n) where n is the length of lst1 + length of lst2
    :Aux space complexity: O(n) where n is the length of lst1 + lst2
    """
    i, j = 0, 0
    lst = []
    while i < len(lst1) and j < len(lst2):
        if lst1[i][index] >= lst2[j][index]:
            lst.append(lst2[j])
            j += 1
        else:
            lst.append(lst1[i])
            i += 1
    
    # adding the leftovers from either lst1 or lst2, takes O(N) time at max
    for every in range(i, len(lst1)):
        lst.append(lst1[every])
    for every in range(j, len(lst2)):
        lst.append(lst2[every])
    return lst 

# Got the code from fit 1045
def merge_sort(ls, index):
    """ Sorts the list of lists in non-decreasing order according to index given

    :Input: 
    ls - a non-empty list of lists 
    index: an integer representing the index

    :Output:
    A list of lists which is sorted in non-decreasing order according to the index 

    :Time complexity: O(nlogn) where n is the length of the ls
    :Aux space complexity: O(n) where n is the length of the ls 
    """
    n = len(ls)
    if n <= 1:
        return ls
    else:
        sub1 = merge_sort(ls[:n//2], index)
        sub2 = merge_sort(ls[n//2:], index)
    return merge(sub1, sub2, index)