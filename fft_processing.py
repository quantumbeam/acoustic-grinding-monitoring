import pandas as pd
import numpy as np
import os
import glob

def calculate_fft_power(file_path, sampling_rate=2e6, header_lines=12,
                        start_sample_index=200013, end_sample_index=1200012): # Added parameters
    """
    Reads an AE data file, applies a Hamming window, performs FFT,
    and calculates the total power in the 100 kHz to 1 MHz range,
    skipping initial noise.

    Args:
        file_path (str): Path to the AE data CSV file.
        sampling_rate (float): Sampling rate in Hz.
        header_lines (int): Number of header lines to skip in the CSV.
        start_sample_index (int): The 0-based index to start reading the actual signal data.
                                  Corresponds to ~0.1s to skip initial noise.
        end_sample_index (int): The 0-based index to stop reading the actual signal data (exclusive).
                                Corresponding to ~0.6s to define the 0.5s effective data.

    Returns:
        float: The total power in the specified frequency band, or None if an error occurs.
    """
    try:
        # 1. Read the signal data
        # The data is in a single column, so we'll read it as a Series.
        # We give it a name 'amplitude' for clarity.
        full_signal = pd.read_csv(
            file_path,
            header=None,
            skiprows=header_lines,
            names=['amplitude'],
            dtype=np.float64
        )['amplitude'].values

        if len(full_signal) == 0:
            print(f"Warning: No data found in {file_path}")
            return None

        # Apply noise skipping based on provided indices
        # Ensure the slice is valid
        if start_sample_index >= len(full_signal) or start_sample_index < 0:
            print(f"Warning: start_sample_index {start_sample_index} out of bounds for {file_path} (length {len(full_signal)})")
            return None
        # end_sample_index can be up to the length of the signal
        effective_end_index = min(end_sample_index, len(full_signal))

        if effective_end_index <= start_sample_index:
             print(f"Warning: effective_end_index {effective_end_index} must be greater than start_sample_index {start_sample_index} for {file_path}")
             return None
        
        # Take the slice of the signal that should be analyzed
        signal = full_signal[start_sample_index:effective_end_index]

        if len(signal) == 0:
            print(f"Warning: No effective data after skipping noise for {file_path}")
            return None

        # 2. Apply Hamming window
        windowed_signal = signal * np.hamming(len(signal))

        # 3. Perform FFT
        N = len(windowed_signal)
        fft_result = np.fft.fft(windowed_signal)

        # 4. Calculate power spectrum
        # Power is the square of the magnitude of the FFT result.
        # We normalize by the number of points.
        power_spectrum = np.abs(fft_result)**2 / N

        # 5. Calculate total power in the 100 kHz to 1 MHz band
        # First, get the frequency bins
        freq_bins = np.fft.fftfreq(N, 1/sampling_rate)

        # We only need the positive frequencies
        positive_freq_mask = freq_bins >= 0
        freqs = freq_bins[positive_freq_mask]
        power = power_spectrum[positive_freq_mask]

        # Find the indices for the desired frequency band
        band_mask = (freqs >= 100e3) & (freqs <= 1e6) # 100 kHz to 1 MHz

        # Sum the power in that band
        total_power = np.sum(power[band_mask])

        return total_power

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

if __name__ == '__main__':
    # --- Configuration ---
    # Path to a sample file to test the function
    SAMPLE_FILE = 'ae_data/exp2/NaCl/1st/20251216_144933NaCl_grind25min.csv'
    SAMPLING_RATE = 2e6 # 2 MHz as specified in the paper
    HEADER_LINES = 12   # As we determined
    START_SAMPLE_INDEX = 200013 # Corresponds to ~0.1s
    END_SAMPLE_INDEX = 1200012 # Corresponds to ~0.6s

    # --- Execution ---
    print(f"Processing file: {SAMPLE_FILE}")

    # Check if the sample file exists
    if not os.path.exists(SAMPLE_FILE):
        print(f"Error: Sample file not found at '{SAMPLE_FILE}'")
        print("Please update the SAMPLE_FILE variable in the script.")
    else:
        # Calculate the total power
        total_ae_power = calculate_fft_power(
            SAMPLE_FILE,
            sampling_rate=SAMPLING_RATE,
            header_lines=HEADER_LINES,
            start_sample_index=START_SAMPLE_INDEX,
            end_sample_index=END_SAMPLE_INDEX
        )

        if total_ae_power is not None:
            print(f"\nTotal AE Power (100kHz - 1MHz): {total_ae_power:.4f}")

    # --- Optional: Process a few files from a directory ---
    print("\n--- Processing a few more examples ---")
    example_files = glob.glob('ae_data/exp2/NaCl/1st/*.csv')[:3] # Get first 3 files

    if not example_files:
        print("No example files found to process.")
    else:
        for file in example_files:
            power = calculate_fft_power(
                file,
                sampling_rate=SAMPLING_RATE,
                header_lines=HEADER_LINES,
                start_sample_index=START_SAMPLE_INDEX,
                end_sample_index=END_SAMPLE_INDEX
            )
            if power is not None:
                print(f"File: {os.path.basename(file)}, Power: {power:.4f}")