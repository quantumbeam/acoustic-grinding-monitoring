"""Referee 2 (comment 1): material-screening indices quoted in Section 4.2.

Two dimensionless quantities are computed from the exp2 (time-based) runs:

1. Initial-to-steady-state AE ratio P_1 / P_33, averaged over the three
   replicates of the 25 min runs.  The denominator uses the same definition as
   the steady-state levels already reported in the manuscript (27.4 / 10.2 /
   260.4), i.e. the final raw acquisition of the run.
2. The exponent alpha of S_AE ~ D50^alpha, fitted in log-log space over the
   seven calibration durations.  S_AE is the final value of the 4-point moving
   average of each run, matching Eq. (2) and Section 3.2; D50 is read from the
   `Dx (50)` row of the PSD CSV, as everywhere else in this repository.
"""

import glob
import json
import os
import re
from datetime import datetime

import numpy as np

from ae_fft import calculate_fft_power

MATERIALS = ["NaCl", "Citricacid", "MSG"]
TRIALS = ["1st", "2nd", "3rd"]
DURATIONS_MIN = [3, 5, 7, 10, 15, 20, 25]
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")


def norm_path(path):
    return os.path.normpath(os.path.abspath(path))


def parse_timestamp(path):
    match = re.search(r"(\d{8}_\d{6})", os.path.basename(path))
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else datetime.min


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def ae_series(cache, material, trial, minutes):
    files = sorted(
        glob.glob(f"data/ae/exp2/{material}/{trial}/*grind{minutes}min*.csv"),
        key=lambda p: (parse_timestamp(p), os.path.basename(p)),
    )
    values = []
    for path in files:
        power = cache.get(norm_path(path))
        if power is None:
            power = calculate_fft_power(path)
        if power is not None:
            values.append(power * 1e6)
    return np.array(values)


def get_d50(material, trial, minutes):
    matches = glob.glob(f"data/powder_size_distribution/exp2/{material}/{trial}/*grind{minutes}min*.csv")
    if not matches:
        return None
    with open(matches[0], "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip().startswith("Dx (50)"):
                return float(line.split(",")[1])
    return None


def main():
    cache = load_cache()

    print("Initial-to-steady-state AE ratio (exp2, 25 min, mean of 3 replicates)")
    print(f"{'material':12s}{'P_1':>10s}{'P_33':>10s}{'ratio':>9s}   per-trial")
    for material in MATERIALS:
        series = [ae_series(cache, material, trial, 25) for trial in TRIALS]
        first = np.array([s[0] for s in series])
        last = np.array([s[-1] for s in series])
        per_trial = np.round(first / last, 1)
        print(f"{material:12s}{first.mean():10.1f}{last.mean():10.1f}"
              f"{first.mean() / last.mean():9.1f}   {per_trial}")

    print()
    print("Power-law sensitivity  S_AE ~ D50^alpha  (7 calibration durations, mean of 3 replicates)")
    print(f"{'material':12s}{'alpha':>8s}{'R^2':>8s}{'AE range':>10s}{'D50 range (um)':>18s}")
    for material in MATERIALS:
        d50_values, s_ae_values = [], []
        for minutes in DURATIONS_MIN:
            sizes = [get_d50(material, trial, minutes) for trial in TRIALS]
            powers = []
            for trial in TRIALS:
                series = ae_series(cache, material, trial, minutes)
                powers.append(series[-4:].mean() if series.size >= 4 else series.mean())
            if any(size is None for size in sizes):
                continue
            d50_values.append(float(np.mean(sizes)))
            s_ae_values.append(float(np.mean(powers)))
        d50_values = np.array(d50_values)
        s_ae_values = np.array(s_ae_values)
        slope, _ = np.polyfit(np.log10(d50_values), np.log10(s_ae_values), 1)
        r = np.corrcoef(np.log10(d50_values), np.log10(s_ae_values))[0, 1]
        print(f"{material:12s}{slope:8.2f}{r ** 2:8.3f}"
              f"{s_ae_values[0] / s_ae_values[-1]:9.1f}x"
              f"{d50_values[0]:12.1f} -> {d50_values[-1]:.1f}")


if __name__ == "__main__":
    main()
