import math

import numpy as np
import cv2


def generic_map(f, img):  # todo: usar lib que gam achou pra implementar isso praticamente igual implementou tranformação linear
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

v: np.ndarray = cv2.imread('assets/cat.jpg')


new_img = generic_map(f, v)

cv2.imwrite('new_img.png', new_img)
