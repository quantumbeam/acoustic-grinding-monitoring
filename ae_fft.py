import pandas as pd
import numpy as np
import os
import glob

def calculate_fft_power(file_path, sampling_rate=2e6, header_lines=12,
                        start_sample_index=200013, end_sample_index=1200012):
    """
    Reads an AE data file (16-bit offset binary), converts it to Voltage,
    performs FFT (without windowing), and calculates the total power 
    for frequencies > 100 kHz.
    
    Returns:
        float: The total power in V^2 (Volts squared).
    """
    try:
        # 1. Read the signal data (Raw 16-bit integer)
        full_signal_raw = pd.read_csv(
            file_path,
            header=None,
            skiprows=header_lines,
            names=['amplitude'],
            dtype=np.float64
        )['amplitude'].values

        if len(full_signal_raw) == 0:
            return None

        # --- 追加: Rawデータ(整数)から電圧(V)への変換 ---
        # ADC仕様: 16bit, Range ±1V, Straight Offset Binary
        # 0 -> -1V, 32768 -> 0V, 65535 -> +1V
        full_signal_v = (full_signal_raw - 32768) / 32768.0

        # Apply noise skipping
        if start_sample_index >= len(full_signal_v) or start_sample_index < 0:
            return None
        effective_end_index = min(end_sample_index, len(full_signal_v))
        if effective_end_index <= start_sample_index:
             return None
        
        # 切り出し
        signal = full_signal_v[start_sample_index:effective_end_index]

        if len(signal) == 0:
            return None

        # 2. No Window Function (Rectangular Window)
        windowed_signal = signal

        # 3. Perform FFT
        N = len(windowed_signal)
        fft_result = np.fft.fft(windowed_signal)

        # 4. Calculate Amplitude Spectrum
        freq_bins = np.fft.fftfreq(N, 1/sampling_rate)
        positive_freq_mask = freq_bins >= 0
        freqs = freq_bins[positive_freq_mask]
        
        # Amplitude = abs(FFT) / N * 2 (for AC components)
        amplitude_spectrum = np.abs(fft_result[positive_freq_mask]) / N * 2

        # 5. Calculate total power > 100 kHz to 1 MHz
        band_mask = (freqs >= 100000) & (freqs <= 1000000)
        target_amplitudes = amplitude_spectrum[band_mask]
        
        # Power in V^2 (Sum of squared amplitudes)
        total_power = np.sum(target_amplitudes**2)

        return total_power

    except FileNotFoundError:
        return None
    except Exception as e:
        # print(f"Error processing {file_path}: {e}")
        return None

if __name__ == '__main__':
    # --- Configuration ---
    # テスト用にあなたがアップロードしたファイルパスに合わせています
    SAMPLE_FILE = '20251219_110041MSG_grind5min.csv' 
    SAMPLING_RATE = 2e6 
    HEADER_LINES = 12   
    START_SAMPLE_INDEX = 200013 
    END_SAMPLE_INDEX = 1200012 

    # --- Execution ---
    print(f"Processing file: {SAMPLE_FILE}")

    if not os.path.exists(SAMPLE_FILE):
        print(f"Error: Sample file not found at '{SAMPLE_FILE}'")
        # 既存のパスがあればそちらで試行
        alt_files = glob.glob('data/ae/exp2/NaCl/1st/*.csv')
        if alt_files:
            SAMPLE_FILE = alt_files[0]
            print(f"Using alternative file: {SAMPLE_FILE}")

    if os.path.exists(SAMPLE_FILE):
        # Calculate Power (returns V^2)
        total_ae_power_v2 = calculate_fft_power(
            SAMPLE_FILE,
            sampling_rate=SAMPLING_RATE,
            header_lines=HEADER_LINES,
            start_sample_index=START_SAMPLE_INDEX,
            end_sample_index=END_SAMPLE_INDEX
        )

        if total_ae_power_v2 is not None:
            # Display in both V^2 and mV^2
            total_ae_power_mv2 = total_ae_power_v2 * 1e6
            print(f"\nTotal AE Power (> 100kHz):")
            print(f"  {total_ae_power_v2:.4e} V²")
            print(f"  {total_ae_power_mv2:.4f} mV²")