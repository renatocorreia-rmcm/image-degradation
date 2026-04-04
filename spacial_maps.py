import math

import numpy as np
import cv2


def linear_map(matrix: np.ndarray, img: np.ndarray):
    assert matrix.shape == (2, 2)
    assert img.ndim == 3

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

    for i in range(h):
        for j in range(w):
            new_coord = matrix @ np.array([i, j]) + corretor

            ni = int(new_coord[0])
            nj = int(new_coord[1])

            new_img[ni][nj] = img[i][j]
            old_img[i-i_min][j-j_min] = img[i][j]

    return old_img, new_img


v: np.ndarray = cv2.imread('assets/gam.jpg')
A = np.array([
    [0.5, 0],
    [-0.5, 0.5]
])

theta = -math.pi / 8
R = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])



def generic_map(f, img):
    """
		map each vector in the original image to its new position using any spacial map
	"""
    assert img.ndim == 3

    new_img = np.zeros(shape=img.shape, dtype='uint8')

    for i, line in enumerate(img):
        for p, pixl in enumerate(line):
            # compute new coordinates
            new_coord = f([p, i])
            # assign to new img
            if 0 <= new_coord[0] < img.shape[1] and 0 <= new_coord[1] < img.shape[0]:
                new_img[int(new_coord[1])][int(new_coord[0])] = pixl

    return new_img

def f(coord):
    return [coord[0] + math.sin(coord[1] / 60) * 70, coord[1]]


old_img, new_img = linear_map(A, v)

cv2.imwrite('old_img.png', old_img)
cv2.imwrite('new_img.png', new_img)
