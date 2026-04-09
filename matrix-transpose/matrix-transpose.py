import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    AT = np.array([np.array([0 for j in range(len(A))]) for i in range(len(A[0]))])
    for i in range(len(A)):
        for j in range(len(A[0])):
            AT[j][i] = A[i][j]

    return(AT)