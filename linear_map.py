import math

import numpy as np
import cv2


def bilerp(v00, v01, v10, v11, di, dj):
    return (
            v00 * (1 - di) * (1 - dj) +
            v01 * (1 - di) * dj +
            v10 * di * (1 - dj) +
            v11 * di * dj
    )


def linear_map(matrix: np.ndarray, img: np.ndarray):  # todo: use our Fl type
    assert matrix.shape == (2, 2)
    assert img.ndim == 3

    assert np.linalg.det(matrix) != 0
    matrix_inv = np.linalg.inv(matrix)  # todo: test gauss elimination and lu decomposition

    h, w = img.shape[0], img.shape[1]

    # vértices no formato (i, j)
    original_vertices = [
        [0, 0],
        [0, w - 1],
        [h - 1, 0],
        [h - 1, w - 1]
    ]

    # aplica transformação
    transformed_vertices = [
        matrix @ np.array(v) for v in original_vertices
    ]

    # calcula boundings
    all_i = [v[0] for v in original_vertices] + [v[0] for v in transformed_vertices]
    all_j = [v[1] for v in original_vertices] + [v[1] for v in transformed_vertices]

    i_min = int(np.floor(np.min(all_i)))
    i_max = int(np.ceil(np.max(all_i)))
    j_min = int(np.floor(np.min(all_j)))
    j_max = int(np.ceil(np.max(all_j)))

    # deslocamento pra evitar índice negativo
    corretor = np.array([
        -i_min if i_min < 0 else 0,
        -j_min if j_min < 0 else 0
    ])

    new_h = i_max - i_min + 1
    new_w = j_max - j_min + 1

    new_img = np.zeros((new_h, new_w, 3), dtype='uint8')
    old_img = np.zeros((new_h, new_w, 3), dtype='uint8')

    for new_i in range(new_h):
        for new_j in range(new_w):
            old_i, old_j = matrix_inv @ np.array([new_i, new_j] - corretor)  # todo: optimize: avoid to transform every vector in black board. Could predict iff it lands on img without doing so.

            if (
                    0 <= old_i < h and
                    0 <= old_j < w
            ):
                # img[old_i][old_j]

                # bounds safety
                old_i = max(0, min(old_i, h - 1))
                old_j = max(0, min(old_j, w - 1))

                fi = int(math.floor(old_i))
                fj = int(math.floor(old_j))
                ci = min(fi + 1, h - 1)
                cj = min(fj + 1, w - 1)

                v00 = img[fi, fj]
                v01 = img[fi, cj]
                v10 = img[ci, fj]
                v11 = img[ci, cj]

                di = old_i - fi
                dj = old_j - fj

                color = bilerp(v00, v01, v10, v11, di, dj)  # todo: implement others interpolations

                new_img[new_i, new_j] = color
                old_img[int(old_i), int(old_j)] = color

    return old_img, new_img


v: np.ndarray = cv2.imread('assets/cat.jpg')

A = np.array([
    [3, 3],
    [0, 3]
])

theta = -math.pi / 8
R = np.array([
    [math.cos(theta), math.sin(theta)],
    [-math.sin(theta), math.cos(theta)]
])


old_img, new_img = linear_map(A, v)

cv2.imwrite('old_img.png', old_img)
cv2.imwrite('new_img.png', new_img)
