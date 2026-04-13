import math
import numpy as np
import cv2
import sympy as sp

from Fl import Fl
from interp import bilerp



def f(coord):
    """
    Função definida com matemática do SymPy.
    """
    return [coord[0] + sp.sin(coord[1] / 30) * (coord[1] / 5), coord[1]]


def create_inverse_function(forward_func):
    """
    Usa o SymPy para encontrar a inversa algébrica de uma função de mapeamento
    e a compila (lambdify) para rodar rápido no Python usando 'math'.
    """

    x_orig, y_orig = sp.symbols('x_orig y_orig')
    x_novo, y_novo = sp.symbols('x_novo y_novo')

    # Calcula o mapeamento simbólico e monta o sistema
    resultado = forward_func([x_orig, y_orig])
    eq1 = sp.Eq(x_novo, resultado[0])
    eq2 = sp.Eq(y_novo, resultado[1])

    # Resolve o sistema e pega a primeira solução
    inversas = sp.solve((eq1, eq2), (x_orig, y_orig), dict=True)
    if not inversas:
        raise ValueError("Não foi possível encontrar uma inversa para a função fornecida.")

    expr_x = inversas[0][x_orig]
    expr_y = inversas[0][y_orig]

    # print(f"Inversa : \n x_orig = {expr_x}\n y_orig = {expr_y}\n")

    # Compila a expressão do SymPy em funções nativas do Python
    calc_x_velho = sp.lambdify((x_novo, y_novo), expr_x, modules='math')
    calc_y_velho = sp.lambdify((x_novo, y_novo), expr_y, modules='math')

    # Retorna o Wrapper final que o seu mapeador vai usar
    def f_inv(coord, fl=False):
        x_new, y_new = coord[0], coord[1]
        x_old = calc_x_velho(x_new, y_new)
        y_old = calc_y_velho(x_new, y_new)

        if fl:
            return [Fl(x_old), Fl(y_old)]
        else:
            return [x_old, y_old]

    return f_inv


def generic_map_interpolated(f_inv, img, fl=False):
    """
    Mapeamento reverso com interpolação bilinear otimizada .
    """
    assert img.ndim == 3
    assert img.shape[2] == 4  # Exige BGRA

    h, w, c = img.shape
    new_img = np.zeros(shape=img.shape, dtype='uint8')

    # Mapeamento Reverso (Destino -> Origem)
    for i in range(h):  # y_novo
        for j in range(w):  # x_novo

            # 1. Busca Reversa
            orig_coords = f_inv([j, i], fl=fl)

            # 2. Extração de valor (híbrido: Fl ou float nativo)
            def get_val(x):
                return float(x.value) if hasattr(x, 'value') else float(x)

            val_j = get_val(orig_coords[0])  # x_orig
            val_i = get_val(orig_coords[1])  # y_orig

            if 0 <= val_i <= h - 1 and 0 <= val_j <= w - 1:

                # Interpolação Bilinear
                new_img[i, j] = bilerp(img=img, old_i=val_i, old_j=val_j, h=h, w=w, fl=fl)
            else:
                # Se cair fora da imagem original, fica transparente
                new_img[i, j] = np.array([0, 0, 0, 0], dtype='uint8')

    return new_img




if __name__ == "__main__":

    v = cv2.imread('assets/cat.jpg')
    v = cv2.cvtColor(v, cv2.COLOR_BGR2BGRA)

    # Gera a função inversa
    f_inv = create_inverse_function(f)

    # Imagem real jogada
    img_normal = generic_map_interpolated(f_inv, v, fl=False)
    cv2.imwrite('cat_normal.png', img_normal)

    # Imagem representável
    img_fl = generic_map_interpolated(f_inv, v, fl=True)
    cv2.imwrite('cat_fl.png', img_fl)



    diff_img = cv2.absdiff(img_normal, img_fl)

    print(np.max(diff_img[:, :, :3]))

    # pixel(i+1) = |pixel(i)*a + b|
    diff_amplificada = cv2.convertScaleAbs(diff_img, alpha=1, beta=0)
    diff_amplificada[:, :, 3] = 255

    cv2.imwrite('cat_diff_amplified.png', diff_amplificada)
