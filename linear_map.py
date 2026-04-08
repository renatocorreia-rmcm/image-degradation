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

def clamp(x, mi = 0, ma = 255):
    return min(ma, max(mi,x))

def rotate(img: np.ndarray, angle: float = (2*np.pi)/360):
    return linear_map(
        matrix=np.array([
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)]
        ]),
        img=img
    )


def resize(img: np.ndarray, factor: float = None, width: int = None, height: int = None):
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
        width = (height/img.shape[0])*img.shape[1]

    return linear_map(
        np.array([
            [height/img.shape[0], 0],
            [0, width/img.shape[1]]
        ]),
        img
    )


def linear_map(matrix:np.ndarray, img:np.ndarray, fl:bool=False, interpolation:str="bilinear"):
    assert matrix.shape == (2, 2)  # R2 square matrix
    assert img.ndim == 3  #  matrix of pixels
    assert img.shape[2] == 4  # alpha channel
    assert interpolation in ("bilinear", "bicubic")

    assert np.linalg.det(matrix) != 0
    matrix_inv = np.linalg.inv(matrix)  # todo: test gauss elimination, lu decomposition and QR decomposition

    h, w = img.shape[0], img.shape[1]

    # vértices no formato (i, j)
    original_vertices = [
        [0, 0],
        [0, w - 1],
        [h - 1, 0],
        [h - 1, w - 1]
    ]

    # aplica transformação
    transformed_vertices = np.array([
        matrix @ np.array(v) for v in original_vertices
    ])

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

    new_img = np.zeros((new_h, new_w, 4), dtype='uint8')
    old_img = np.zeros((new_h, new_w, 4), dtype='uint8')

    for new_i in range(
            int(np.floor(np.min(transformed_vertices[:, 0]))+corretor[0]),
            int(np.ceil(np.max(transformed_vertices[:, 0]))+corretor[0]) + 1
    ):
        for new_j in range(
                int(np.floor(np.min(transformed_vertices[:, 1]))+corretor[1]),
                int(np.ceil(np.max(transformed_vertices[:, 1]))+corretor[1]) + 1
        ):
            old_i, old_j = matrix_inv @ np.array([new_i, new_j] - corretor)

            if (
                    0 <= old_i < h and
                    0 <= old_j < w
            ):  # todo: else: transparente (ao inves de preto) (adicionar canal alpha na imagem)
                # img[old_i][old_j]

                # bounds safety
                old_i = max(0, min(old_i, h - 1))
                old_j = max(0, min(old_j, w - 1))

                if interpolation == 'bicubic':
                    color = interp.bicubic(img, old_i, old_j, h, w, fl)
                else:
                    fi, fj = int(math.floor(old_i)), int(math.floor(old_j))
                    ci, cj = min(fi + 1, h - 1), min(fj + 1, w - 1)

                    v00, v01 = img[fi, fj], img[fi, cj]
                    v10, v11 = img[ci, fj], img[ci, cj]

                    di = old_i - fi
                    dj = old_j - fj

                    color = interp.bilerp(v00, v01, v10, v11, di, dj, fl)

                new_img[new_i, new_j] = color

    vc = np.vectorize(clamp)

    for i, line in enumerate(img):
        for j, col in enumerate(line):
            old_img[i + corretor[0], j + corretor[1]] = vc(img[i, j])

    return old_img, new_img


if __name__ == "__main__":

    v: np.ndarray = load_img('assets/tinycat.jpg')

    A = np.array([
        [1, 0],
        [0, 1]
    ])

    v_fl = mtx.to_fl_matrix(v)
    _, v_fl_new = linear_map(A, v_fl, fl=True, interpolation="bicubic")
    
    _, v_new_bicubic = linear_map(A, v, interpolation="bicubic")
    _, v_new_linear = linear_map(A, v)
    error = abs(v_fl_new.astype(np.float64) - v_new_linear.astype(np.float64))

    error_norm = (error / error.max()) * 255 if error.max() > 0 else error
    error_norm = error_norm.astype(np.uint8)

    error  = error.astype(np.uint8)

    print(f"mse: {mtx.mse(v_fl_new, v_new_linear)}")

    cv2.imwrite("error_norm.jpeg", error_norm)
    cv2.imwrite("error.jpeg", error)
    cv2.imwrite("fl.png", v_fl_new)
    cv2.imwrite("noFl.png", v_new_linear)
    cv2.imwrite("resize_bicubic.png", v_new_bicubic)
    cv2.imwrite("resize_bilinear.png", v_new_linear)