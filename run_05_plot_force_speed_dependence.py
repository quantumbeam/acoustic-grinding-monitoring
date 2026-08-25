import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import sys
from datetime import datetime
from ae_fft import calculate_fft_power
import scienceplots
plt.style.use(['science', 'ieee', 'no-latex'])


# ================== 設定 ==================
experiment_name_list = ["force", "spr"]
png_output_dir = os.path.join("analysis_results", "run_05_plot_force_speed_dependence")
# 移動平均の点数
moving_average_window = 4
# FFT processing parameters (match ae_fft defaults)
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

def get_experiment_config(name):
    # region ディレクトリのパスをリストで指定
    # --- フォース違い ---
    if name == "force":
        dir_path1 = "data/ae/discussion/force/10N"
        dir_path2 = "data/ae/discussion/force/20N"
        dir_paths = [dir_path1, dir_path2]
        labels = ["10 N", "20 N"]
    # --- spr違い ---
    elif name == "spr":
        dir_path1 = "data/ae/discussion/speed/spr0_5"
        dir_path2 = "data/ae/discussion/speed/spr1"
        dir_path3 = "data/ae/discussion/speed/spr1_5"
        dir_paths = [dir_path1, dir_path2, dir_path3]
        labels = ["0.5 s per cycle", "1.0 s per cycle", "1.5 s per cycle"]
    else:
        print("experiment_nameを正しく設定してください。")
        sys.exit(1)
    # endregion ==================================
    return dir_paths, labels


# --- グラフの体裁 ---
# ラベル
# Reviewer 1 (technical comment 3): the plotted quantity is the AE acquisition
# number (one measurement per 24 grinding cycles), not the number of motions.
# For the force experiment every series uses the same 0.5 s cycle period, so a
# net-grinding-time axis is well defined (24 x 0.5 s = 12 s per acquisition).
# For the speed experiment the cycle period differs between series
# (12 / 24 / 36 s of net grinding per acquisition), so no single time axis can
# be overlaid and the acquisition number is kept as the only x axis.
GRINDING_CYCLES_PER_ACQUISITION = 24
FORCE_CYCLE_PERIOD_S = 0.5
FORCE_NET_GRINDING_S_PER_ACQUISITION = (
    GRINDING_CYCLES_PER_ACQUISITION * FORCE_CYCLE_PERIOD_S
)
TIME_X_LABEL = "Grinding time (min)"
ACQUISITION_X_LABEL = "AE acquisition number"
ACQUISITION_TICK_STEP = 5


def parse_timestamp(file_path):
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(file_path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def elapsed_minutes(file_paths, net_grinding_s=FORCE_NET_GRINDING_S_PER_ACQUISITION):
    """Elapsed process time (min) of each acquisition, from the file timestamps."""
    stamps = [parse_timestamp(p) for p in file_paths]
    if any(t is None for t in stamps):
        return np.arange(1, len(file_paths) + 1) * net_grinding_s / 60.0
    t0 = stamps[0]
    offsets = np.array([(t - t0).total_seconds() for t in stamps])
    return (offsets + net_grinding_s) / 60.0
y_label = r"Total spectral power (a.u.)"

# フォントサイズ
font_size = 24
label_font_size = 32
tick_font_size = 24

# ---凡例---
legend_font_size = 18
# ==========================================

def plot_fft_power_for_multiple_dirs(directory_paths, output_plot_path, x_label, y_label,labels=None,
                                     use_time_axis=False):
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
            if use_time_axis:
                # Measured elapsed process time of each acquisition (Reviewer 1,
                # technical comment 3). Valid moving-average points start at the
                # window-th acquisition.
                time_values = elapsed_minutes(csv_files)[moving_average_window - 1:]
            else:
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

    ax = plt.gca()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if use_time_axis:
        ax.set_xlim(left=0.0)
    ax.legend()
    plt.tight_layout()

    # プロットをファイルに保存
    plt.savefig(output_plot_path)
    print(f"プロットを '{output_plot_path}' に保存しました。")

if __name__ == "__main__":
    for experiment_name in experiment_name_list:
        dir_paths, labels = get_experiment_config(experiment_name)
        name_for_file = "speed" if experiment_name == "spr" else experiment_name
        output_file = os.path.join(png_output_dir, f"fft_power_trend_{name_for_file}.png")

        if dir_paths:
            # Both figures use the measured elapsed process time. For the
            # speed experiment each series has its own timeline, because the
            # cycle period (and hence the duration of a 24-cycle block) differs
            # between series.
            plot_fft_power_for_multiple_dirs(
                dir_paths,
                output_file,
                TIME_X_LABEL,
                y_label,
                labels=labels,
                use_time_axis=True,
            )
