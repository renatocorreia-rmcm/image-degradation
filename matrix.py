import numpy as np
from Fl import Fl

def to_fl_matrix(M: np.ndarray) -> np.ndarray:
    return np.vectorize(Fl)(M)

def LU_factorization(matrix:np.ndarray, fl=False, pivoting=True) -> tuple[np.array]:
    size = matrix.shape[0]

    convert = to_fl_matrix if fl else lambda x: x.astype(np.float64)

    lower = convert(np.eye(size))
    upper = convert(matrix.copy())
    permutation = convert(np.eye(size))

    for i in range(size):
        if pivoting:
            row = i
            pivo = abs(upper[i][i])

            for k in range(i, size):
                if abs(upper[k][i]) > pivo:
                    pivo = abs(upper[k][i])
                    row = k

            if row != i:
                upper[[row, i]] = upper[[i, row]]
                permutation[[row, i]] = permutation[[i, row]]

                for j in range(i):
                    lower[row][j], lower[i][j] = lower[i][j], lower[row][j]

        pivo = upper[i][i]

        if pivo == 0:
            raise Exception("Impossible to LU decompose the given matrix")

        for j in range(i+1, size):
            ml = upper[j][i] / pivo

            lower[j][i] = ml
            upper[j] -= upper[i] * ml

    return permutation, lower, upper

# PA = UL -> A = P⁻¹·UL -> A⁻¹ = (P⁻¹·UL)⁻¹ = L⁻¹U⁻¹·P 
def inverse_matrix(matrix:np.ndarray, fl=False) -> np.ndarray:
    # Ly = B
    # Ux = y
    size = np.shape(matrix)[0]

    inverse = np.zeros((size, size))
    if fl: inverse = to_fl_matrix(inverse)
    
    try:
        permutation, lower, upper = LU_factorization(matrix, fl=fl, pivoting=True)
    except Exception:
        raise Exception("Not inversible")

    for i in range(size):
        b = np.zeros(size)
        b[i] = 1

        # Ly = b
        y = np.zeros(size)
        if fl: y = to_fl_matrix(y)
        for m in range(size):
            y[m] = (b[m] - sum([lower[m][n]*y[n] for n in range(m)])) / lower[m][m]

        # Ux = y
        x = np.zeros(size)
        if fl: x = to_fl_matrix(x)
        for m in range(size-1, -1, -1):
            x[m] = (y[m] - sum([upper[m][n]*x[n] for n in range (m+1, size)])) / upper[m][m]

        for j in range(size):
            inverse[j][i] = x[j]
    
    return inverse @ permutation