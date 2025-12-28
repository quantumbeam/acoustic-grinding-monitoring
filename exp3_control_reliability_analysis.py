import joblib
import pandas as pd
import numpy as np
import os

# --- 出力先設定 ---
os.makedirs('results', exist_ok=True)

# 1. 生データ入力
raw_measured = {
    'NaCl': {250: [223.6, 252.1, 238.2], 200: [196.1, 204.4, 189.5], 150: [133.4, 123.8, 139.8]},
    'CitricAcid': {100: [83.2, 100.2, 111.6], 50: [46.1, 31.4, 37.3], 20: [21.3, 14.1, 16.9]},
    'Ajinomoto': {200: [179.6, 203.3, 189.0], 100: [129.3, 124.9, 123.9], 50: [53.1, 54.9, 50.1]}
}

stop_ae_power = {
    'NaCl': {250: [1.3e-5, 1.2e-5, 1.25e-5], 200: [8.8e-6, 9.1e-6, 8.9e-6], 150: [5.8e-6, 5.6e-6, 5.9e-6]},
    'CitricAcid': {100: [2.5e-6, 2.7e-6, 2.4e-6], 50: [1.1e-6, 1.0e-6, 1.2e-6], 20: [4.5e-7, 4.8e-7, 4.4e-7]},
    'Ajinomoto': {200: [3.4e-5, 3.6e-5, 3.3e-5], 100: [1.8e-5, 1.7e-5, 1.9e-5], 50: [7.8e-6, 8.0e-6, 7.6e-6]}
}

# 2. モデルファイル名マッピング（ご提示のファイル名に完全一致）
model_files = {
    'NaCl': 'results/gpr_model_NaCl_exp2.joblib',
    'CitricAcid': 'results/gpr_model_Citricacid_exp2.joblib',
    'Ajinomoto': 'results/gpr_model_Ajinomoto_exp2.joblib'
}

stats_list = []
uncertainty_list = []

for material, targets in raw_measured.items():
    # モデルのロード
    path = model_files[material]
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"✅ Loaded: {path}")
    else:
        print(f"❌ Not Found: {path}")
        model = None

    for target_val, trials in targets.items():
        # --- A. 実測統計 (exp3_raw_stats.csv) ---
        trials = np.array(trials)
        mean_val = np.mean(trials)
        sigma_trial = np.std(trials, ddof=1)
        stats_list.append({
            'Material': material, 'Target_um': target_val,
            'Measured_Mean_um': round(mean_val, 2),
            'Sigma_Trial_um': round(sigma_trial, 2),
            'Error_um': round(mean_val - target_val, 2)
        })

        # --- B. GPR不確かさ (exp3_gpr_uncertainty.csv) ---
        if model is not None:
            powers = np.array(stop_ae_power[material][target_val]).reshape(-1, 1)
            y_pred, y_sigma = model.predict(powers, return_std=True)
            uncertainty_list.append({
                'Material': material, 'Target_um': target_val,
                'GPR_Pred_Mean_um': round(np.mean(y_pred), 2),
                'GPR_Sigma_Uncertainty_um': round(np.mean(y_sigma), 2)
            })

# 3. 保存
pd.DataFrame(stats_list).to_csv('results/exp3_raw_stats.csv', index=False)
pd.DataFrame(uncertainty_list).to_csv('results/exp3_gpr_uncertainty.csv', index=False)

print("\n--- Process Finished ---")
print("Saved: results/exp3_raw_stats.csv (Actual trials variance)")
print("Saved: results/exp3_gpr_uncertainty.csv (AI model uncertainty)")