import pandas as pd
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
# 変更点1: ConstantKernel を追加
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# --- 1. あなたから提供された全生データ ---
raw_data_map = {
    'NaCl': {
        'targets': [250, 200, 150],
        'measured': [[223.6, 252.1, 238.2], [196.1, 204.4, 189.5], [133.4, 123.8, 139.8]],
        'ae_power': [[1.3e-5, 1.2e-5, 1.25e-5], [8.8e-6, 9.1e-6, 8.9e-6], [5.8e-6, 5.6e-6, 5.9e-6]]
    },
    'CitricAcid': {
        'targets': [100, 50, 20],
        'measured': [[83.2, 100.2, 111.6], [46.1, 31.4, 37.3], [21.3, 14.1, 16.9]],
        'ae_power': [[2.5e-6, 2.7e-6, 2.4e-6], [1.1e-6, 1.0e-6, 1.2e-6], [4.5e-7, 4.8e-7, 4.4e-7]]
    },
    'Ajinomoto': {
        'targets': [200, 100, 50],
        'measured': [[179.6, 203.3, 189.0], [129.3, 124.9, 123.9], [53.1, 54.9, 50.1]],
        'ae_power': [[3.4e-5, 3.6e-5, 3.3e-5], [1.8e-5, 1.7e-5, 1.9e-5], [7.8e-6, 8.0e-6, 7.6e-6]]
    }
}

os.makedirs('results', exist_ok=True)
stats_list = []      # 実測統計用 (exp3_raw_stats.csv)
uncertainty_list = [] # GPR推論用 (exp3_gpr_uncertainty.csv)

print("--- Starting Complete Analysis (Stats & GPR) ---")

for material, content in raw_data_map.items():
    # --- A. 実測統計の計算 (Trial-to-Trial) ---
    for i, target_val in enumerate(content['targets']):
        measured_trials = np.array(content['measured'][i])
        mean_measured = np.mean(measured_trials)
        sigma_trial = np.std(measured_trials, ddof=1)
        
        stats_list.append({
            'Material': material,
            'Target_um': target_val,
            'Measured_Mean_um': round(mean_measured, 2),
            'Sigma_Trial_um': round(sigma_trial, 2),
            'Error_um': round(mean_measured - target_val, 2)
        })

    # --- B. 逆モデル(GPR)の学習と推論 ---
    X_train = np.array(content['ae_power']).flatten().reshape(-1, 1) # AE Power
    y_train = np.array(content['measured']).flatten()                # D50
    
    # 変更点2: カーネル定義をGitHub実装 (..._gauss.py) に合わせる
    # ConstantKernelを使用し、初期値は1.0、bounds指定を削除（デフォルトを使用）
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) \
             + WhiteKernel(noise_level=1.0)
    
    # 変更点3: random_state=0 を削除
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_train, y_train)
    
    for i, target_val in enumerate(content['targets']):
        test_powers = np.array(content['ae_power'][i]).reshape(-1, 1)
        y_pred, y_sigma = gpr.predict(test_powers, return_std=True)
        
        uncertainty_list.append({
            'Material': material,
            'Target_um': target_val,
            'GPR_Pred_Mean_um': round(np.mean(y_pred), 2),
            'GPR_Sigma_Uncertainty_um': round(np.mean(y_sigma), 2)
        })

# --- 2. 2つのCSVに分けて保存 ---
pd.DataFrame(stats_list).to_csv('results/exp3_raw_stats.csv', index=False)
pd.DataFrame(uncertainty_list).to_csv('results/exp3_gpr_uncertainty.csv', index=False)

print("\n✅ Analysis Finished Successfully.")
print("1. 'results/exp3_raw_stats.csv'       <- 実測の再現性(Table用)")
print("2. 'results/exp3_gpr_uncertainty.csv' <- モデルの確信度(Discussion用)")