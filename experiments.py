from linear_map import load_img, linear_map
import matrix as mtx
import numpy as np
import interp
import cv2
import os


def save_results(results, filename):

    dir_name = os.path.splitext(filename)[0]
    output_path = os.path.join("experiments_linear", dir_name)

    os.makedirs(output_path, exist_ok=True)

    for k, methods in results.items():
        for interp_method, img_data in methods.items():
            
            file_name = f"{k}_{interp_method}.png"
            full_file_path = os.path.join(output_path, file_name)
            
            success = cv2.imwrite(full_file_path, img_data)
            
            if success:
                print(f"Salvo: {full_file_path}")
            else:
                print(f"Erro ao salvar: {full_file_path}")


def run_linear_experiment(filename: str, A:np.ndarray, n: int, interp_methods: np.ndarray):

    v = load_img(f'assets/{filename}')

    v_fl = mtx.to_fl_matrix(v)

    results = {
        method: {
            'fl': v_fl.copy(), 
            'no_fl': v.copy()
        } for method in interp_methods
    }


    for i in range(n):

        print(f"\nIteration : {i+1}")

        for interp_method, machines in results.items():
            for machine in machines.keys():
                
                print(f"{interp_method} -- {machine}")

                func = getattr(interp, interp_method)
                _, new_img = linear_map(A, results[interp_method][machine], fl=(machine == 'fl'), interpolation=func)

                results[interp_method][machine] = new_img

    return results

# Reflection 
A = np.array([
    [1, 0],
    [0, -1]
])
n = 1 # sequence length

img = "tinycat.jpg"

results = run_linear_experiment(img, A, n, ["lanczos"])

save_results(results, img)



