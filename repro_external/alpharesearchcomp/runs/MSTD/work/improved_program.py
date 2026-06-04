import numpy as np


def main():
    N = 30

    # Exact span-normalized search found this MSTD set with
    # |A + A| = 51 and |A - A| = 47.
    A = [0, 1, 2, 4, 5, 9, 12, 13, 17, 20, 21, 22, 24, 25]
    B = A[:]

    A_ind = np.zeros(N, dtype=int)
    B_ind = np.zeros(N, dtype=int)
    A_ind[A] = 1
    B_ind[B] = 1
    return A_ind, B_ind


try:
    A_indicators
    B_indicators
except NameError:
    A_indicators, B_indicators = main()
