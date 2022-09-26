"""
Author: Satya Jhaveri
Unit: FIT2004
Purpose: Test Cases for FIT2004 Assignment 3, 2022 Semester 1
Disclaimer: This contains the test cases from the Assignment pdf, and some custom ones I added.
    I can't guarantee that the custom test cases are all right, but afaik they're correct, pls
    point out any errors tho
    
    Also just because your code has 0 errors doesn't mean you'll get 100% on your assignment, check 
    your code fits the time and space complexity requirements listed in the instructions, both 
    tasks have pretty (relatively) easy-to-code brute force solutions but that ain't what the markers
    are looking for.

To use this tester: Just put this file in the same directory as your assignment code file and run this 
    file(your code file must be named "assignment3.py", as per the assignment instructions).
"""

from assignment3 import best_revenue, hero
import unittest

class TestLocalMaximum(unittest.TestCase):
    def setUp(self) -> None:
        # Create list of errors to collect over all tests:
        self.errorList = []
    
    def tearDown(self) -> None:
        # Print all errors found:
        for e in self.errorList:
            print(e)
        print(f"Number of errors: {len(self.errorList)}")
    
    def test_best_revenue(self) -> None:
        # implementing my own test cases since the fit2004 teaching team are MEAN and won't give us ANY :((
        # Test set 1:
        #         city:  0   1   2    # city:
        travel_days = [[0,  1,  1],  # 0
                       [ 1, 0,  1],  # 1
                       [ 1,  2, 0]]  # 2
        #     city: 0  1    2     # day:
        revenue = [[1, 2,   1],   # 0
                   [3, 3,   1],   # 1
                   [1, 1, 100]]   # 2
        start = 0
        expected_max_profit = 101
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")
            
            
        #         city:  0   1   2    # city:
        travel_days = [[0,  1,  1],  # 0
                       [ 1, 0,  1],  # 1
                       [ 1,  2, 0]]  # 2
        #     city: 0  1    2     # day:
        revenue = [[1, 2,   1],   # 0
                   [3, 3,   1],   # 1
                   [1, 1, 100]]   # 2
        start = 1
        expected_max_profit = 102
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")
        
        
        #         city:  0   1   2    # city:
        travel_days = [[0,  1,  1],  # 0
                       [ 1, 0,  1],  # 1
                       [ 1,  2, 0]]  # 2
        #     city: 0  1    2     # day:
        revenue = [[1, 2,   1],   # 0
                   [3, 3,   1],   # 1
                   [1, 1, 100]]   # 2
        start = 2
        expected_max_profit = 102
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")
        
        
        
        # Test set 2:
        #       city:    0  1  2  3   # city:
        travel_days = [[0,-1, 3, 1], # 0
                       [-1,0,-1, 1], # 1
                       [ 1,-1,0, 1], # 2
                       [ 1, 1, 2,0]] # 3
        
        #       city:    0     1  2  3 # day:
        revenue =     [[ 1,    2, 3, 4], # 0
                       [ 3,    6, 1, 5], # 1
                       [ 1,    8, 4, 1], # 2
                       [ 1,   10, 4, 5], # 3
                       [10,    4, 5, 9]] # 4
        start = 0
        expected_max_profit = 22
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")
            
            
        #       city:    0  1  2  3   # city:
        travel_days = [[0,-1, 3, 1], # 0
                       [-1,0,-1, 1], # 1
                       [ 1,-1,0, 1], # 2
                       [ 1, 1, 2,0]] # 3
        
        #       city:    0     1  2  3 # day:
        revenue =     [[ 1,    2, 3, 4], # 0
                       [ 3,    6, 1, 5], # 1
                       [ 1,    8, 4, 1], # 2
                       [ 1,   10, 4, 5], # 3
                       [10,    4, 5, 9]] # 4
        start = 1
        expected_max_profit = 30
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")


        #       city:    0  1  2  3   # city:
        travel_days = [[0,-1, 3, 1], # 0
                       [-1,0,-1, 1], # 1
                       [ 1,-1,0, 1], # 2
                       [ 1, 1, 2,0]] # 3
        
        #       city:    0     1  2  3 # day:
        revenue =     [[ 1,    2, 3, 4], # 0
                       [ 3,    6, 1, 5], # 1
                       [ 1,    8, 4, 1], # 2
                       [ 1,   10, 4, 5], # 3
                       [10,    4, 5, 9]] # 4
        start = 2
        expected_max_profit = 22
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")
        
        
        #       city:    0  1  2  3   # city:
        travel_days = [[0,-1, 3, 1], # 0
                       [-1,0,-1, 1], # 1
                       [ 1,-1,0, 1], # 2
                       [ 1, 1, 2,0]] # 3
        
        #       city:    0     1  2  3 # day:
        revenue =     [[ 1,    2, 3, 4], # 0
                       [ 3,    6, 1, 5], # 1
                       [ 1,    8, 4, 1], # 2
                       [ 1,   10, 4, 5], # 3
                       [10,    4, 5, 9]] # 4
        start = 3
        expected_max_profit = 28
        try:
            res = best_revenue(revenue, travel_days, start)
            self.assertEqual(res, expected_max_profit, msg=f"Output for best_revenue(revenue={revenue}, travel_days={travel_days}, start={start})\n incorrect. \nExpected:{expected_max_profit}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"best_revenue(revenue={revenue}, travel_days={travel_days}, start={start}) raised the error: {str(e)}")


    def test_hero(self) -> None:
        # implementing the test cases from assignment3.pdf:
        # Example 01:
        attacks =[[1,2,7,5], [2,1,4,4], [3,6,9,2]]
        expected_res = sorted([[3,6,9,2], [2,1,4,4]])  # sorting since order is irrelevant and its easier to equate with function output then
        try:
            res = sorted(hero(attacks))
            self.assertEqual(res, expected_res, msg=f"Output for hero(attacks={attacks}) incorrect. \nExpected:{expected_res}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"hero(attacks={attacks}) raised the error: {str(e)}")
        
        # Example 02:
        attacks = [[1,2,7,5], [2,1,4,4], [3,5,6,2]]
        expected_res = sorted([[3,5,6,2], [2,1,4,4]])
        try:
            res = sorted(hero(attacks))
            self.assertEqual(res, expected_res, msg=f"Output for hero(attacks={attacks}) incorrect. \nExpected:{expected_res}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"hero(attacks={attacks}) raised the error: {str(e)}")
        
        # Example 03:
        attacks = [[1,2,7,6], [2,7,9,10], [3,8,9,5]]
        expected_res = sorted([[3,8,9,5], [1,2,7,6]])
        try:
            res = sorted(hero(attacks))
            self.assertEqual(res, expected_res, msg=f"Output for hero(attacks={attacks}) incorrect. \nExpected:{expected_res}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"hero(attacks={attacks}) raised the error: {str(e)}")
        
        # adding my own test cases:
        attacks = [[1,1,3,9], [2,3,4,5], [3,1,5,2], [4,4,6,8], [5,3,9,1], [6,5,8,8], [7,5,10,6], [8,8,11,5]]
        expected_res = sorted([[8,8,11,5], [4,4,6,8], [1,1,3,9]])
        try:
            res = sorted(hero(attacks))
            self.assertEqual(res, expected_res, msg=f"Output for hero(attacks={attacks}) incorrect. \nExpected:{expected_res}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"hero(attacks={attacks}) raised the error: {str(e)}")
        
        # All starting at the same time:
        attacks = [[1,0,2,4], [2,0,6,12], [3,0,53,6], [4,0,2,6], [5,0,8,2], [6,0,3,5]]
        expected_res = [[2,0,6,12]]
        try:
            res = sorted(hero(attacks))
            self.assertEqual(res, expected_res, msg=f"Output for hero(attacks={attacks}) incorrect. \nExpected:{expected_res}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"hero(attacks={attacks}) raised the error: {str(e)}")
        
        # All ending at the same time:
        attacks = [[1,1,100,7], [2,4,100,8], [3,23,100,12], [4,31,100,6], [5,51,100,9]]
        expected_res = [[3,23,100,12]]
        try:
            res = sorted(hero(attacks))
            self.assertEqual(res, expected_res, msg=f"Output for hero(attacks={attacks}) incorrect. \nExpected:{expected_res}.\nYour function returned: {res}\n")
        except AssertionError as e:
            self.errorList.append(str(e))
        except Exception as e:
            self.errorList.append(f"hero(attacks={attacks}) raised the error: {str(e)}")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLocalMaximum)
    unittest.TextTestRunner(verbosity=0).run(suite)