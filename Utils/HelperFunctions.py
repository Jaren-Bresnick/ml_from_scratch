from itertools import combinations_with_replacement
import numpy as np

# Polynomial Regression
def expand_polynomial_features(X, degree):
    feature_indices = range(X.shape[1])
    actual_combinations = []
    combs_replacement = combinations_with_replacement(feature_indices, degree)
    for combination in combs_replacement:
        feature = np.ones(X.shape[0])
        for index in combination:
            feature *= X[ :, index]
        actual_combinations.append(feature)
    stacked_features = np.column_stack(actual_combinations)
    return stacked_features