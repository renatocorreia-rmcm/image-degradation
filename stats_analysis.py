import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
from error_analysis import plot_error_boxplot, plot_error_histogram

import cv2

path = "experiments_linear/tinycat_shear"
methods = ["bilerp", "bicubic", "lanczos"]
metrics = ["MSE", "RMSE", "MAE", "PSNR", "SSIM", "Delta_E"]

dfs = {}
for m in methods:
    file_path = os.path.join(path, f"stats_{m}.csv")
    if os.path.exists(file_path):
        dfs[m] = pd.read_csv(file_path)
    else:
        print(f"Aviso: Arquivo {file_path} não encontrado.")

scaler = MinMaxScaler()

for method, df in dfs.items():

    df_metrics = df[metrics].copy()

    # Min-Max Normalization
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_metrics),
        columns=metrics,
        index=df.index
    )

    if 'Tempo' in df.columns:
        df_normalized['Tempo'] = df['Tempo'].str.extract('(\d+)').astype(int)
    else:
        df_normalized['Tempo'] = df.index

    plt.figure(figsize=(12, 6))

    df_plot = df_normalized.melt(id_vars='Tempo', var_name='Métrica', value_name='Valor Normalizado')

    sns.lineplot(data=df_plot, x='Tempo', y='Valor Normalizado', hue='Métrica', marker='o')

    plt.title(f'Métricas Normalizadas (Min-Max) - Método: {method}')
    plt.xlabel('Iteração / Tempo')
    plt.ylabel('Valor Normalizado (0-1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f'experiments_linear/tinycat_shear/grafico_{method}.png', bbox_inches='tight')
    print(f'Gráfico para {method} salvo com sucesso.')
    plt.close()


    matriz_fl = cv2.imread(img1_fl_path, cv2.IMREAD_COLOR)
    matriz_no_fl = cv2.imread(img1_no_fl_path, cv2.IMREAD_COLOR)
plot_error_boxplot()
