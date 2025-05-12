from itertools import combinations_with_replacement
import numpy as np

# Polynomial Regression
def expand_polynomial_features(X, degree):
    actual_combinations = []
    combinations = combinations_with_replacement(X, degree)
    for combination in combinations:
        total = 1
        for value in combination:
            total *= value
        actual_combinations.append(total)
    stacked_features = np.hstack(actual_combinations)
    
    return stacked_features