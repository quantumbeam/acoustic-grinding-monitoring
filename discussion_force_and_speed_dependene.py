import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
from fft_processing import calculate_fft_power
import datetime
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])


# ================== 設定 ==================
experiment_name_list=["force","spr","Other"]
experiment_name="spr"
png_output_dir = "results/discussion/force_and_speed_dependene"
# 移動平均の点数
moving_average_window = 4
# FFT processing parameters (match fft_processing defaults)
sampling_rate = 2e6
header_lines = 12
start_sample_index = 200013
end_sample_index = 1200012
os.makedirs(png_output_dir, exist_ok=True)
# 手動でパスを設定
dir_path1="csv_data/ae_data_csv/20251219/grind25min"
dir_path2="csv_data/ae_data_csv/20251219_2nd/grind25min"
dir_paths=[dir_path1,dir_path2]
labels=["Data 1","Data 2"]

# region ディレクトリのパスをリストで指定
# --- フォース違い ---
if experiment_name=="force":
    dir_path1="ae_data/discussion/force/10N"
    dir_path2="ae_data/discussion/force/20N"
    dir_paths=[dir_path1,dir_path2]
    labels=["10 N","20 N"]
# --- spr違い ---
elif experiment_name=="spr":
    dir_path1="ae_data/discussion/speed/spr0_5"
    dir_path2="ae_data/discussion/speed/spr1"
    dir_path3="ae_data/discussion/speed/spr1_5"
    dir_paths=[dir_path1,dir_path2,dir_path3]
    labels=["0.5 s per cycle","1.0 s per cycle","1.5 s per cycle"]

# --- その他 ---
elif experiment_name=="Other":
    pass

else:
    print("experiment_nameを正しく設定してください。")
    sys.exit(1)
# endregion ==================================


# --- グラフの体裁 ---
# ラベル
x_label = "Number of motions"
y_label = r"Total Power Spectrum($\mathrm{mV}^2$)"

# フォントサイズ
font_size = 24
label_font_size = 32
tick_font_size = 24

# ---凡例---
legend_font_size = 18
# ==========================================

def plot_fft_power_for_multiple_dirs(directory_paths, output_plot_path, x_label, y_label,labels=None):
    if labels is None:
        labels = [f"Data {i+1}" for i in range(len(directory_paths))]
    markers = ['o', 'x', '^', 's', 'D', 'v', '<', '>']
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = label_font_size
    plt.rcParams['xtick.labelsize'] = tick_font_size
    plt.rcParams['ytick.labelsize'] = tick_font_size
    plt.rcParams['legend.fontsize'] = legend_font_size

    for i, directory_path in enumerate(directory_paths):
        csv_pattern = os.path.join(directory_path, '*.csv')
        csv_files = sorted(glob.glob(csv_pattern))

        if not csv_files:
            print(f"ディレクトリ '{directory_path}' にCSVファイルが見つかりません。")
            continue

        power_values = []
        for csv_path in csv_files:
            print(f"処理中のファイル: {csv_path}")
            power = calculate_fft_power(
                csv_path,
                sampling_rate=sampling_rate,
                header_lines=header_lines,
                start_sample_index=start_sample_index,
                end_sample_index=end_sample_index,
            )
            if power is not None:
                power_values.append(power)
            else:
                print(f"警告: {csv_path} の処理中にエラーが発生しました。")

        if power_values:
            
            parts = directory_path.split("/")
            dir_name = "/".join(parts[-2:])
            power_values = np.array(power_values)
            # 移動平均を計算
            power_values_move_ave = np.convolve(power_values, np.ones(moving_average_window)/moving_average_window, mode='valid')
            time_values = np.arange(len(power_values_move_ave))+moving_average_window
            # print(time_values)
            # plt.plot(time_values, power_values_move_ave*10**6, marker='o', linestyle='-', label=dir_name)
            plt.plot(time_values, power_values_move_ave*10**6, marker=markers[i], label=labels[i])
            # # フォース違い
            # if i==0:
            #     plt.plot(time_values, power_values_move_ave*10**6, marker='o',label="10 N")
            # elif i==1:
            #     plt.plot(time_values, power_values_move_ave*10**6, marker='x',label="20 N")
            # #spr違い
            # if i==0:
            #     plt.plot(time_values, power_values_move_ave*10**6,marker="o",label="0.5 s per rotation")
            # elif i==1:
            #     plt.plot(time_values, power_values_move_ave*10**6,marker="x",label="1.0 s per rotation")
            # elif i==2:
            #     plt.plot(time_values, power_values_move_ave*10**6,marker="^",label="1.5 s per rotation")
            
        else:
            print(f"ディレクトリ '{directory_path}' で有効なFFTパワー値を計算できませんでした。")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    # プロットをファイルに保存
    plt.savefig(output_plot_path)
    print(f"プロットを '{output_plot_path}' に保存しました。")

if __name__ == "__main__":
    # 日付から保存パスを作成
    now = datetime.datetime.now()
    
    output_file =png_output_dir +"/"+  f"{now.strftime('%Y%m%d_%H%M%S')}_fft_power_trend.pdf"
    target_directories = dir_paths
    
    if target_directories:
        plot_fft_power_for_multiple_dirs(target_directories, output_file, x_label, y_label,labels=labels)
