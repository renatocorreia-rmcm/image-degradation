
import numpy as np
import cv2

import matplotlib.pyplot as plt
import seaborn as sns


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


def get_statistics(filename_fl: str, filename_no_fl:str):
    img_fl_uint = cv2.imread(filename_fl, cv2.IMREAD_COLOR)
    img_no_fl_uint = cv2.imread(filename_no_fl, cv2.IMREAD_COLOR)

    img_fl = img_fl_uint.astype(np.float64)
    img_no_fl = img_no_fl_uint.astype(np.float64)


    diff = cv2.absdiff(img_fl, img_no_fl)

    plot_error_histogram(diff)

    cv2.imwrite("diff_test.png", diff)
    
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


    
    print("--- Estatísticas ---")
    print(f"Max Error (L_inf): {max_err}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"PSNR: {psnr} dB")
    print(f"SSIM: {ssim_val}")
    print(f"Mean Delta E: {mean_delta_e}\n")
    
    plot_error_histogram(diff)


    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Max_Error": max_err,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "Delta_E": mean_delta_e
    }


# To do: criar uma função que itere sobre as imagens em experiments linear
#        pegue duas imagens do msm método de interpolaçao e armazene as estatísticas em um csv
# - talvez fazer essa análise a cada iteração da sequencia de trasnformações seja melhor para analisar
# Nesse caso, essa análise de erro rodaria dentro do run_linear_experiment (lembrar de tirar do plot do histograma, caso for fazer assim)
if __name__ == "__main__":
    get_statistics("experiments_linear/tinycat/lanczos_fl.png", "experiments_linear/tinycat/lanczos_no_fl.png")
