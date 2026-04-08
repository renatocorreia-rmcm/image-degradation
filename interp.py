import numpy as np 
import math
import matrix as mtx

def clamp(x, mi = 0, ma = 255):
    return min(ma, max(mi,x))

def bilerp(v00, v01, v10, v11, di, dj, fl=False):
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
        clamp(int(rgb[0])),
        clamp(int(rgb[1])),
        clamp(int(rgb[2])),
        clamp(int(alpha * 255))
    ], dtype=np.uint8)

def cubic_kernel(t, a=-0.75):
    t = abs(t)
    if t <= 1:
        return (a + 2) * t**3 - (a + 3) * t**2 + 1
    elif t < 2:
        return a * t**3 - 5*a * t**2 + 8*a * t - 4*a
    return 0

def bicubic(img, old_i, old_j, h, w, fl=False):
    fi = int(math.floor(old_i))
    fj = int(math.floor(old_j))
    di = old_i - fi
    dj = old_j - fj

    acc_rgb   = np.zeros(3, dtype=np.float64)
    if fl: acc_rgb = mtx.to_fl_matrix(acc_rgb)

    acc_alpha = 0.0
    total_weight = 0.0

    for m in range(-1, 3):       # 4 linhas
        wi = cubic_kernel(m - di)
        for n in range(-1, 3):   # 4 colunas
            wj = cubic_kernel(n - dj)
            w_total = wi * wj

            pi = clamp(fi + m, 0, h - 1)
            pj = clamp(fj + n, 0, w - 1)
            p  = img[pi, pj]

            alpha = p[3] / 255.0
            if alpha > 0:
                acc_rgb   += p[:3] * w_total * alpha
                acc_alpha += w_total * alpha
                total_weight += w_total * alpha

    if total_weight > 0:
        rgb   = acc_rgb / total_weight
        alpha = acc_alpha / (sum(
            cubic_kernel(m - di) * cubic_kernel(n - dj)
            for m in range(-1, 3) for n in range(-1, 3)
        ))
    else:
        return np.array([0, 0, 0, 0], dtype=np.uint8)

    return np.array([
        clamp(int(rgb[0])),
        clamp(int(rgb[1])),
        clamp(int(rgb[2])),
        clamp(int(alpha * 255))
    ], dtype=np.uint8)