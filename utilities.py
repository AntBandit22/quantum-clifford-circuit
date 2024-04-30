# List of utility functions for specific calculations

import numpy as np


def g_function(x_1, z_1, x_2, z_2):
    # function that needs to be defined to multiply rows together with abstract definition
    # Uses for row_sum method in Tableau class
    # Inputs range from 0 to 1
    
    check_list = [x_1, z_1]
    x, y, z = [1,0], [1,1], [0,1]
    if check_list == x:
        return z_2 * (2 * x_2 -1)
    elif check_list == y:
        return z_2 - x_2
    elif check_list == z:
        return x_2 * (1 - 2 * z_2)
    else:
        return 0
    
def count_paulis(row_array:np.ndarray, active_row: int):
    # Used in sort_active_region method in Tableau class
    # Counts the number of different paulis in a column.
    # Input is a 1D numpy array containing the integers representing the Pauli matrices.
    # Input array should be sorted.
    # 1 = X, 2 = Z, 3 = Y, 4 = I
    # active_row is an integer.
    # Example:
    # row_array = [1,2,3,4] -> 3 different Paulis -> returns 3

    # Assertion checks
    assert type(active_row) == int, f"active_row should be an integer!"

    count = 0
    if row_array[active_row] == 4:
        return count
    else:
        count = 1
        for element in range(len(row_array)):
            if row_array[element] != 4:
                if (element > active_row) & (row_array[element] != row_array[element - 1]):
                    count += 1
    return count

def check_commutation(row_a, row_b):
    # Input two Pauli Strings in binary form
    # Outputs 0 if they commute
    # Outputs 1 if they anticommute
    # Used in Measurement, and calculating Entanglement Negativity
    
    commutation_sum = 0
    for position in range(int(len(row_a)/2)):
        commutation_sum += row_a[2*position]*row_b[2*position+1] + row_a[2*position+1]*row_b[2*position]
    return commutation_sum%2

