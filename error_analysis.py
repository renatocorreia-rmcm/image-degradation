
import numpy as np
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_error_histogram(diff):
    colors = ('b', 'g', 'r')
    labels = ('Blue Channel', 'Green Channel', 'Red Channel')

    plt.figure(figsize=(10, 6))

    for i, col in enumerate(colors):
        sns.histplot(
            diff[:, :, i].ravel(), 
            bins=50, 
            kde=True, 
            color=col, 
            label=labels[i],
            alpha=0.3,      
            element="step"  
        )
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title("Error Distribution: Histogram + KDE (per Channel)")
    plt.xlabel("Error Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_error_boxplot(diff):
    # Flatten + estrutura "long format"
    data = pd.DataFrame({
        "Error": np.concatenate([
            diff[:, :, 0].ravel(),
            diff[:, :, 1].ravel(),
            diff[:, :, 2].ravel()
        ]),
        "Channel": (
            ["Blue"] * diff[:, :, 0].size +
            ["Green"] * diff[:, :, 1].size +
            ["Red"] * diff[:, :, 2].size
        )
    })

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=data,
        x="Channel",
        y="Error",
        width=0.5,
        showfliers=True,
        fliersize=2,
        linewidth=1.5
    )

    # Linha de erro zero
    plt.axhline(0, linestyle='--', linewidth=1.2)

    plt.title("Error Distribution per Channel")
    plt.xlabel("Color Channel")
    plt.ylabel("Error")

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.show()

def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # Constantes do artigo original do SSIM (Wang et al.)
    # Usadas para evitar divisão por zero quando a variância é muito baixa
    K1 = 0.01
    K2 = 0.03
    L = 255.0 # Valor máximo do pixel
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Converter para float64 para não perder decimais no meio da conta
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    # 1. Calcular as médias locais usando um filtro Gaussiano 11x11
    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 2. Calcular as variâncias e covariâncias locais
    sigma1_sq = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2

    # 3. Fórmula principal do SSIM
    numerador = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominador = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerador / denominador

    # Retorna a média de todos os pixels e de todos os canais (RGB)
    return float(np.mean(ssim_map))


def get_statistics(img_fl_input: np.ndarray, img_no_fl_input: np.ndarray):
    img_fl_uint = np.clip(np.round(img_fl_input), 0, 255).astype(np.uint8)
    img_no_fl_uint = np.clip(np.round(img_no_fl_input), 0, 255).astype(np.uint8)

    img_fl = img_fl_input.astype(np.float64)
    img_no_fl = img_no_fl_input.astype(np.float64)


    diff = cv2.absdiff(img_fl, img_no_fl)

    # cv2.imwrite("diff_test.png", diff)
    
    # Max Absolute Error
    max_err = np.max(diff)

    # Mean Squared Error
    mse = round(np.mean(diff ** 2), 4)

    # Root Mean Squared Error
    rmse = round(np.sqrt(mse), 4)

    # Mean Absolute Error
    mae = round(np.mean(np.abs(diff)), 4)

    # PSNR (Peak Signal-to-Noise Ratio)
    if rmse == 0:
        psnr = float('inf')
    else:
        psnr = round(cv2.PSNR(img_fl_uint, img_no_fl_uint), 4)

    # Structural Similarity Index
    ssim_val = round(ssim(img_fl_uint, img_no_fl_uint), 4)


    # Delta E
    lab_fl = cv2.cvtColor(img_fl_uint, cv2.COLOR_BGR2LAB).astype(np.float64)
    lab_no_fl = cv2.cvtColor(img_no_fl_uint, cv2.COLOR_BGR2LAB).astype(np.float64)
    
    delta_e = np.sqrt(np.sum((lab_fl - lab_no_fl) ** 2, axis=2))
    mean_delta_e = round(np.mean(delta_e), 4)


    
    # print("--- Estatísticas ---")
    # print(f"Max Error (L_inf): {max_err}")
    # print(f"RMSE: {rmse}")
    # print(f"MAE: {mae}")
    # print(f"PSNR: {psnr} dB")
    # print(f"SSIM: {ssim_val}")
    # print(f"Mean Delta E: {mean_delta_e}\n")
    
    # plot_error_histogram(diff)
    # plot_error_boxplot(diff)


    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Max_Error": max_err,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "Delta_E": mean_delta_e
    }


if __name__ == "__main__":

    img1_fl_path = "experiments_linear/tinycat/lanczos_fl.png"
    img1_no_fl_path = "experiments_linear/tinycat/lanczos_no_fl.png"

    matriz_fl = cv2.imread(img1_fl_path, cv2.IMREAD_COLOR)
    matriz_no_fl = cv2.imread(img1_no_fl_path, cv2.IMREAD_COLOR)

    print(get_statistics(matriz_fl,matriz_no_fl))
