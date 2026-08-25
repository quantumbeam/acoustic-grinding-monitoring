"""Referee 1 (scientific comment 3): lattice parameters vs the plateau S_AE.

The reviewer asks whether the steady-state (plateau) AE level is governed by the
crystal lattice.  This script builds the requested comparison table by combining

* the measured AE levels of the 25 min runs (mean of the three replicates),
  using the same definitions as elsewhere in this repository - the initial level
  is the first raw acquisition and the steady-state level is the final raw
  acquisition, i.e. the 215.3 / 1838.3 / 3453.1 and 27.4 / 10.2 / 260.4 mV^2
  quoted in Section 4.2 - with the last-4-point mean given as a more robust
  alternative; and
* crystallographic data taken from the literature.

IMPORTANT: the crystallographic values below are literature values entered by
hand.  They must be checked against CSD/ICSD before submission, and the form of
the citric acid reagent (anhydrous vs monohydrate) must be confirmed from the
container or by TGA/PXRD - the values below assume the anhydrous form.
"""

import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ae_fft import calculate_fft_power

MATERIALS = ["NaCl", "Citricacid", "MSG"]
TRIALS = ["1st", "2nd", "3rd"]
GRIND_MIN = 25
PLATEAU_POINTS = 4
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ae_power_cache.json")
OUTPUT_DIR = os.path.join("analysis_results", "run_09_compare_lattice_and_plateau")

# --- Literature crystallographic data (VERIFY against CSD/ICSD) ---------------
CRYSTAL_DATA = {
    "NaCl": {
        "compound": "Sodium chloride (halite)",
        "crystal_system": "cubic",
        "space_group": "Fm-3m (225)",
        "lattice_parameters_A": "a = 5.6402",
        "Z": 4,
        "habit": "cubic, perfect {100} cleavage",
        "structure_reference": "R. W. G. Wyckoff, Crystal Structures, 2nd ed., Vol. 1, Interscience, 1963 (a = 5.6402 A at 298 K)",
    },
    "Citricacid": {
        "compound": "Citric acid, anhydrous",
        "crystal_system": "monoclinic",
        "space_group": "P2_1/a (14)",
        "lattice_parameters_A": "a = 12.817, b = 5.628, c = 11.465, beta = 111.22 deg",
        "Z": 4,
        "habit": "prismatic",
        "structure_reference": "J. P. Glusker, J. A. Minkin, A. L. Patterson, Acta Crystallogr. B, 1969, 25, 1066-1072 (COD 5000063)",
    },
    "MSG": {
        "compound": "Monosodium L-glutamate monohydrate",
        "crystal_system": "orthorhombic",
        "space_group": "P2_12_12_1 (19)",
        "lattice_parameters_A": "a = 5.521, b = 15.130, c = 17.958",
        "Z": 8,
        "habit": "elongated / needle-like (microscopy, this work)",
        "structure_reference": "C. Sano, N. Nagashima, T. Kawakita, Y. Iitaka, Anal. Sci., 1989, 5, 121-122; cell as quoted in Phys. Chem. Chem. Phys., 2017, 19, 28647",
    },
}
DISPLAY_NAME = {"NaCl": "NaCl", "Citricacid": "Citric acid", "MSG": "MSG"}


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


def ae_series(cache, material, trial):
    files = sorted(
        glob.glob(f"data/ae/exp2/{material}/{trial}/*grind{GRIND_MIN}min*.csv"),
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache = load_cache()

    rows = []
    for material in MATERIALS:
        series = [ae_series(cache, material, trial) for trial in TRIALS]
        series = [s for s in series if s.size]
        initial = np.array([s[0] for s in series])
        steady = np.array([s[-1] for s in series])
        plateau = np.array([s[-PLATEAU_POINTS:].mean() for s in series])

        info = CRYSTAL_DATA[material]
        rows.append(
            {
                "material": DISPLAY_NAME[material],
                "compound": info["compound"],
                "crystal_system": info["crystal_system"],
                "space_group": info["space_group"],
                "lattice_parameters_A": info["lattice_parameters_A"],
                "Z": info["Z"],
                "habit": info["habit"],
                "initial_S_AE_mV2": round(float(initial.mean()), 1),
                "steady_state_S_AE_mV2": round(float(steady.mean()), 1),
                "steady_state_S_AE_sd_mV2": round(float(steady.std(ddof=1)), 1),
                "plateau_last4_S_AE_mV2": round(float(plateau.mean()), 1),
                "plateau_last4_S_AE_sd_mV2": round(float(plateau.std(ddof=1)), 1),
                "initial_to_steady_ratio": round(float(initial.mean() / steady.mean()), 1),
                "n_runs": len(series),
                "structure_reference": info["structure_reference"],
            }
        )

    frame = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "lattice_vs_plateau_S_AE.csv")
    frame.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print()

    columns = [
        "material",
        "crystal_system",
        "space_group",
        "lattice_parameters_A",
        "habit",
        "steady_state_S_AE_mV2",
        "plateau_last4_S_AE_mV2",
    ]
    print(frame[columns].to_string(index=False))
    print()
    print("Ordering check (does the plateau follow lattice symmetry?)")
    ordered = frame.sort_values("steady_state_S_AE_mV2")
    print("  steady-state S_AE, ascending: "
          + " < ".join(f"{r.material} ({r.steady_state_S_AE_mV2})" for r in ordered.itertuples()))
    print("  symmetry, descending:         cubic (NaCl) > orthorhombic (MSG) > monoclinic (Citric acid)")
    print("  -> the highest-symmetry material sits in the middle; no monotonic relation.")


if __name__ == "__main__":
    main()
