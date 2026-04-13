from linear_map import load_img, linear_map
from error_analysis import get_statistics
import matrix as mtx
import numpy as np
import interp
import cv2
import os
import pandas as pd


def save_results(results, statistics, filename):

    dir_name = os.path.splitext(filename)[0]
    output_path = os.path.join("experiments_linear", dir_name)

    os.makedirs(output_path, exist_ok=True)

    # Salvar as Imagens
    for k, methods in results.items():
        for interp_method, img_data in methods.items():
            
            file_name = f"{k}_{interp_method}.png"
            full_file_path = os.path.join(output_path, file_name)
            
            success = cv2.imwrite(full_file_path, img_data)
            
            if success:
                print(f"Salvo: {full_file_path}")
            else:
                print(f"Erro ao salvar: {full_file_path}")

    # Salvar estatísticas
    col_names = ["MSE", "RMSE", "MAE", "Max_Error", "PSNR", "SSIM", "Delta_E"]

    for interp_method, matrix_data in statistics.items():
        df = pd.DataFrame(matrix_data, columns=col_names)
        
        df.index = [f"Iteração {i+1}" for i in range(len(df))]
        df.index.name = "Tempo" 
        
        csv_file_name = f"stats_{interp_method}.csv"
        csv_full_path = os.path.join(output_path, csv_file_name)
        

        df.to_csv(csv_full_path, index=True)


def run_linear_experiment(filename: str, A:np.ndarray, n: int, interp_methods: np.ndarray):

    v = load_img(f'assets/{filename}')
    v_fl = mtx.to_fl_matrix(v)

    results = {
        method: {
            'fl': v_fl.copy(), 
            'no_fl': v.copy()
        } for method in interp_methods
    }
    
    # 7 = # metrics
    stats_serie = {k: np.zeros((n,7)) for k in interp_methods}

    vertices_track = {
        method: {
            'fl': None,
            'no_fl': None
        } for method in interp_methods
    }

    vert= None
    for i in range(n):

        print(f"\nIteration : {i+1}")

        for interp_method, machines in results.items():
            for machine in ['fl', 'no_fl']:
                
                print(f"{interp_method} -- {machine}")

                func = getattr(interp, interp_method)

                v_atual = vertices_track[interp_method][machine]

                new_img, new_vert = linear_map(
                    matrix=A,
                    img=results[interp_method][machine],
                    use_fl=(machine == 'fl'),
                    interpolation=func,
                    vertices_pixels=v_atual
                )

                results[interp_method][machine] = new_img
                vertices_track[interp_method][machine] = new_vert

            cv2.imwrite("fl.png", results[interp_method]['fl'])
            cv2.imwrite("no_fl.png", results[interp_method]['no_fl'])

            stats_serie[interp_method][i] = list(get_statistics(results[interp_method]['fl'], results[interp_method]['no_fl']).values())
    

    return results, stats_serie

# Shear
A = np.array([
    [1, 0.05],
    [0, 1]
])
n = 50 # sequence length

img_path = "tinycat.jpg"

results, statistics = run_linear_experiment(img_path, A, n, ["bicubic"])

save_results(results, statistics, img_path)


