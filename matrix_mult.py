
import numpy as np
from Fl import Fl


def to_fl_matrix(M: np.ndarray) -> np.ndarray:
    return np.vectorize(Fl)(M)

def scalar_prod(v1: np.ndarray, v2: np.ndarray) -> float:
    # v1 : vector 1 x p
    # v2.T : vector n x 1

    # p=n

    if (len(v1) != len(v2)):
        raise ValueError("Vetores devem ter o mesmo tamanho para o produto escalar...")

    resultado= Fl(0)

    for i in range(len(v1)):
        # Fl(ai * bi)
        produto= v1[i] * v2[i]
        
        # Fl(acc + produto)
        resultado= resultado + produto

    return resultado


def matrix_mult(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    # M1 : matriz a1 x b1
    # M2 : matriz a2 x b2

    # M3 : matriz a1 x b2


    a1, b1 = M1.shape
    a2, b2 = M2.shape

    M1_Fl = to_fl_matrix(M1)
    M2_Fl = to_fl_matrix(M2)

    if (b1!=a2):
        print("matrizes não multiplicáveis")
        return 
    
    M3_Fl = np.zeros((a1, b2), dtype=object)

    for i in range(a1):
        for j in range(b2):

            aij = scalar_prod(M1_Fl[i , :] , M2_Fl[:, j])
            M3_Fl[i, j] = aij

    return M3_Fl


# M1 = np.array([(1,1,1),
#                (2,2,2),
#                (3,3,3)])
# M2 = np.array([(1,1,1),
#                (2,2,2),
#                (3,3,3)])
# 
# matrix_mult(M1,M2)
