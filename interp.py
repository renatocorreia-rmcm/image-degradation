import numpy as np 
import math
import matrix as mtx

def _clamp(x, mi = 0, ma = 255):
    return min(ma, max(mi,x))

def bilerp(img, old_i, old_j, h, w, fl=False):
    fi = int(math.floor(old_i))
    fj = int(math.floor(old_j))
    di = old_i - fi
    dj = old_j - fj

    ci, cj = min(fi + 1, h - 1), min(fj + 1, w - 1)

    v00, v01 = img[fi, fj], img[fi, cj]
    v10, v11 = img[ci, fj], img[ci, cj]

    # weights
    w00 = (1 - di) * (1 - dj)
    w01 = (1 - di) * dj
    w10 = di * (1 - dj)
    w11 = di * dj

    pixels = [v00, v01, v10, v11]
    weights = [w00, w01, w10, w11]

    acc_rgb = np.zeros(3, dtype=np.float64)
    if fl: acc_rgb = mtx.to_fl_matrix(acc_rgb)

    acc_alpha = 0.0
    total_weight = 0.0

    for p, w in zip(pixels, weights):
        alpha = p[3] / 255.0

        if alpha > 0:  # ignore fully transparent pixels
            acc_rgb += p[:3] * w * alpha
            acc_alpha += w * alpha
            total_weight += w * alpha

    if total_weight > 0:
        rgb = acc_rgb / total_weight
        alpha = acc_alpha
    else:
        return np.array([0, 0, 0, 0], dtype=np.uint8)

    return np.array([
        _clamp(int(rgb[0])),
        _clamp(int(rgb[1])),
        _clamp(int(rgb[2])),
        _clamp(int(alpha * 255))
    ], dtype=np.uint8)


def bicubic(img, old_i, old_j, h, w, fl=False):
    def _cubic_kernel(t, a=-0.75):
        t = abs(t)
        if t <= 1:
            return (a + 2) * t**3 - (a + 3) * t**2 + 1
        elif t < 2:
            return a * t**3 - 5*a * t**2 + 8*a * t - 4*a
        return 0
        
    fi = int(math.floor(old_i))
    fj = int(math.floor(old_j))
    di = old_i - fi
    dj = old_j - fj

    acc_rgb   = np.zeros(3, dtype=np.float64)
    if fl: acc_rgb = mtx.to_fl_matrix(acc_rgb)

    acc_alpha = 0.0
    total_weight = 0.0

    for m in range(-1, 3):       # 4 linhas
        wi = _cubic_kernel(m - di)
        for n in range(-1, 3):   # 4 colunas
            wj = _cubic_kernel(n - dj)
            w_total = wi * wj

            pi = _clamp(fi + m, 0, h - 1)
            pj = _clamp(fj + n, 0, w - 1)
            p  = img[pi, pj]

            alpha = p[3] / 255.0
            if alpha > 0:
                acc_rgb   += p[:3] * w_total * alpha
                acc_alpha += w_total * alpha
                total_weight += w_total * alpha

    if total_weight > 0:
        rgb   = acc_rgb / total_weight
        alpha = acc_alpha / (sum(
            _cubic_kernel(m - di) * _cubic_kernel(n - dj)
            for m in range(-1, 3) for n in range(-1, 3)
        ))
    else:
        return np.array([0, 0, 0, 0], dtype=np.uint8)

    return np.array([
        _clamp(int(rgb[0])),
        _clamp(int(rgb[1])),
        _clamp(int(rgb[2])),
        _clamp(int(alpha * 255))
    ], dtype=np.uint8)


def lanczos(img, old_i, old_j, h, w, fl=False, a=3):
    """Interpolação Lanczos com janela 2a x 2a"""

    def _lanczos_kernel(t, a=3):
        """Lanczos kernel com janela de tamanho a (padrão a=3)"""
        if t == 0:
            return 1.0
        if abs(t) >= a:
            return 0.0
        pt = math.pi * t
        return a * math.sin(pt) * math.sin(pt / a) / (pt * pt)

    fi = int(math.floor(old_i))
    fj = int(math.floor(old_j))
    di = old_i - fi
    dj = old_j - fj

    acc_rgb      = np.zeros(3, dtype=np.float64)
    acc_alpha    = 0.0
    total_weight = 0.0
    weight_sum   = 0.0

    if fl: acc_rgb = mtx.to_fl_matrix(acc_rgb)

    for m in range(-a + 1, a + 1):        # 2a linhas
        wi = _lanczos_kernel(m - di, a)
        for n in range(-a + 1, a + 1):    # 2a colunas
            wj = _lanczos_kernel(n - dj, a)
            w_total = wi * wj
            weight_sum += w_total

            pi = _clamp(fi + m, 0, h - 1)
            pj = _clamp(fj + n, 0, w - 1)
            p  = img[pi, pj]

            alpha = p[3] / 255.0
            if alpha > 0:
                acc_rgb      += p[:3] * w_total * alpha
                acc_alpha    += w_total * alpha
                total_weight += w_total * alpha

    if total_weight > 0:
        rgb   = acc_rgb / total_weight
        alpha = acc_alpha / weight_sum
    else:
        return np.array([0, 0, 0, 0], dtype=np.uint8)

    return np.array([
        _clamp(int(rgb[0])),
        _clamp(int(rgb[1])),
        _clamp(int(rgb[2])),
        _clamp(int(alpha * 255))
    ], dtype=np.uint8)