import math
import numpy as np
import cv2
import matrix as mtx
import Fl
import interp


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def clamp(x, mi=0, ma=255):
    return min(ma, max(mi, x))


def rotate(img: np.ndarray, angle: float = (2 * np.pi) / 360, vertices_pixels=None):
    return linear_map(
        matrix=np.array([
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)]
        ]),
        img=img,
        vertices_pixels=vertices_pixels
    )


def resize(img: np.ndarray, factor: float = None, width: int = None, height: int = None, vertices_pixels=None):
    """
    takes scaling factor
    or raw new size
    can pass only width or only height and missing parameter will follow img scale
    if pass both width and height, deforms img to that size
    """

    assert factor or width or height  # must give at least one size reference

    if factor is not None:
        assert factor > 0  # resizing takes positive scaling factor
        return linear_map(
            np.array([
                [factor, 0],
                [0, factor]
            ]),
            img
        )

    if not height:
        height = (width / img.shape[1]) * img.shape[0]
    if not width:
        width = (height / img.shape[0]) * img.shape[1]

    return linear_map(
        np.array([
            [height / img.shape[0], 0],
            [0, width / img.shape[1]]
        ]),
        img, vertices_pixels=vertices_pixels
    )


def linear_map(matrix: np.ndarray,
               img: np.ndarray,
               vertices_pixels: np.ndarray = None,
               interpolation=interp.knn,
               use_fl: bool = False):
    """
    parameters as p  refeers to original      space
    parameters as p_ refeers to transformated space

    maps new pixel to old pixel through vector space
    p_ -> p
    is done by
    p_ -> v_ -> v -> p
    """
    assert matrix.shape == (2, 2)  # R2 square matrix
    assert img.ndim == 3  # matrix of pixels
    assert img.shape[2] == 4  # alpha channel

    assert np.linalg.det(matrix) != 0
    matrix_inv = np.linalg.inv(matrix)

    h, w = img.shape[0], img.shape[1]

    if vertices_pixels is None:  # imagem enquadrada perfeitamente
        vertices_pixels = np.array([
            [0, 0],
            [0, w - 1],
            [h - 1, 0],
            [h - 1, w - 1]
        ])

    def get_p(v):
        """
        map v to p
        """
        (a, b) = vertices_pixels[0]
        # p = v + (a, b)
        p = np.array([v[0] + a, v[1] + b])
        return p

    def get_v(p):
        """
        map v to p
        """
        (a, b) = vertices_pixels[0]
        # v = p - (a, b)
        v = np.array([p[0] - a, p[1] - b])
        return v

    vertices_vectors = np.array([get_v(p) for p in vertices_pixels])

    vertices_vectors_ = np.array([
        matrix @ np.array(v) for v in vertices_vectors
    ])

    i_min_vectors_ = np.min(vertices_vectors_[:, 0])
    i_max_vectors_ = np.max(vertices_vectors_[:, 0])
    j_min_vectors_ = np.min(vertices_vectors_[:, 1])
    j_max_vectors_ = np.max(vertices_vectors_[:, 1])

    def get_p_(v_):
        """
        map v to p
        """
        (a_, b_) = [-i_min_vectors_, -j_min_vectors_]  # (0, 0) pra imagem enquadrada
        # p_ = v_ + (a_, b_)
        p_ = np.array([v_[0] + a_, v_[1] + b_])
        return p_

    vertices_pixels_ = np.array([
        get_p_(v_) for v_ in vertices_vectors_
    ])

    h_ = int(i_max_vectors_ - i_min_vectors_) + 1
    w_ = int(j_max_vectors_ - j_min_vectors_) + 1

    img_ = np.zeros((h_, w_, 4), dtype='uint8')

    def get_v_(p_):
        """
        map p_ to v_
        """
        (a_, b_) = [i_min_vectors_, j_min_vectors_]
        # v_ = p_ - (a_, b_)
        v_ = np.array([p_[0] + a_, p_[1] + b_])
        return v_

    def f(p_):
        """
        map p_ to p
        """
        v_ = get_v_(p_)
        v = matrix_inv @ v_
        p = get_p(v)
        return p

    for i_ in range(h_):
        for j_ in range(w_):

            p_ = np.array([i_, j_])
            p = f(p_)

            if (
                    0 <= p[0] < h and
                    0 <= p[1] < w
            ):
                color = interpolation(img, p[0], p[1], h, w, use_fl)

                img_[i_, j_] = color

    return img_, vertices_pixels_


if __name__ == "__main__":
    v: np.ndarray = load_img('assets/tinycat.jpg')

    A = np.array([
        [2, 0],
        [0, 2]
    ])

    v_fl = mtx.to_fl_matrix(v)
    v_fl_new, vertices = linear_map(A, v, use_fl=True, interpolation=interp.knn)

    cv2.imwrite("fl.png", v_fl_new)
