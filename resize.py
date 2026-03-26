import math

import numpy as np
import cv2
from numpy import interp


def resize(old_img: np.ndarray, new_w: int, new_h: int):
	"""
		no interpolation - repeats pixels
	"""
	old_w = old_img.shape[1]
	old_h = old_img.shape[0]

	new_img = np.zeros(shape=([new_h, new_w, 3]), dtype='uint8')

	for new_i, line in enumerate(new_img):
		for new_j, pixl in enumerate(line):

			# discover equivalent coordinate in old image
			old_j = int(new_j*(old_w/new_w))
			old_i = int(new_i*(old_h/new_h))

			new_img[new_i][new_j] = old_img[old_i][old_j]

	return new_img


def grid_interpolate(vertices: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
	x0 = vertices[0][0]


def resize_interp(old_img: np.ndarray, new_w: int, new_h: int):
	"""
		interpolate empty pixels
	"""
	old_w = old_img.shape[1]
	old_h = old_img.shape[0]

	new_img = np.zeros(shape=([new_h, new_w, 3]), dtype='uint8')
	for new_i, line in enumerate(new_img):
		for new_j, pixl in enumerate(line):

			# discover equivalent coordinate in old image
			old_j = new_j*(old_w/new_w)
			old_i = new_i*(old_h/new_h)

			# interpolate (old_i, old_j) inside
			# (floor(old_i), floor(old_j))  (floor(old_i), ceil(old_j))
			# (ceil(old_i), floor(old_j))  (ceil(old_i), ceil(old_j))

			vertices = np.array([
				(math.floor(old_i), math.floor(old_j)), (math.floor(old_i), math.ceil(old_j)),
				(math.ceil(old_i), math.floor(old_j)), (math.ceil(old_i), math.ceil(old_j))
			])
			values = np.array([
				old_img[vertices[0][0]][vertices[0][1]], old_img[vertices[1][0]][vertices[1][1]],
				old_img[vertices[2][0]][vertices[2][1]], old_img[vertices[3][0]][vertices[3][1]],
			])
			target = np.array([old_i, old_j])

			#new_img[new_i][new_j] = grid_interpolate(vertices=vertices, values=values, target=target)

	return new_img


img = cv2.imread('assets/gradiente.png')
cv2.imshow('resized', resize_interp(img, new_w=7, new_h=7))
cv2.waitKey(0)
cv2.destroyAllWindows()

