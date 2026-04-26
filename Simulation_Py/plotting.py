"""
Overlay plot: digitised reference trace vs. simulated FRA trace.

Handles two file formats:
  Format 1 (digitised, e.g. WebPlotDigitizer CSV):
      Comma-separated, no header.
      e.g.  10.444709824196202, 0.038471177944863655

  Format 2 (simulator output):
      Tab-separated, with header 'Frequency(Hz)\tMagnitude(dB)'.
      e.g.  1.000000e+01    -1.695820e-05

Usage:
    python plot_overlay.py
    -> Edit FILE_REF and FILE_SIM paths below to point to your files.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# CONFIG -- EDIT THE PATHS YOU NEED
# ==========================================
FILE_REF = "Simulation_Py\MY_Results.txt"        # digitised reference (Format 1)
FILE_SIM = "Simulation_Py\Wang_Results.txt"     # Python simulator (Format 2)
FILE_LTS = "Simulation_Py\ltspice_results.txt"      # LTSpice AC export (Format 3) - optional, set to None to skip

LABEL_REF = "Wang et al. 2009 (Digitised)"
LABEL_SIM = "This work (Python solver)"
LABEL_LTS = "LTSpice (same circuit)"

OUTPUT_PNG = "Plot_Overlay_Comparison.png"


# ==========================================
# FILE LOADERS (FORMAT-AGNOSTIC)
# ==========================================
def load_trace(filepath):
    """
    Robustly loads a 2-column trace file and returns (frequencies, magnitudes_dB).
    Handles three formats:
      - Format 1: comma/whitespace, no header (digitised, e.g. WebPlotDigitizer)
      - Format 2: tab-separated with 'Frequency(Hz)' header (simulator output)
      - Format 3: LTSpice .txt export, e.g.
            Freq.   V(n028)/V(n012)
            1.0e+01 (-2.79e-03dB,-1.13e-01°)
        Tab-separated, magnitude+phase in parentheses.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")

    with open(filepath, "r", encoding="utf-8-sig") as f:
        raw_lines = f.readlines()

    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    if not lines:
        raise ValueError(f"File {filepath} is empty.")

    # ---- LTSpice format detection ----
    # If any data line contains '(' and 'dB', treat as LTSpice complex export.
    sample_data_line = lines[1] if len(lines) > 1 else lines[0]
    if "(" in sample_data_line and "dB" in sample_data_line:
        return _load_ltspice(lines)

    # ---- Otherwise: generic delimited file ----
    candidates = ["\t", ";", ","]
    sample = lines[: min(20, len(lines))]
    scores = {d: sum(ln.count(d) for ln in sample) for d in candidates}
    best = max(scores, key=scores.get)
    delim = best if scores[best] > 0 else None

    def _is_numeric_row(line):
        parts = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
        if len(parts) < 2:
            return False
        try:
            float(parts[0]); float(parts[1])
            return True
        except ValueError:
            return False

    skip = 0 if _is_numeric_row(lines[0]) else 1

    data = None
    try:
        data = np.genfromtxt(filepath, delimiter=delim, skip_header=skip,
                             encoding="utf-8-sig", autostrip=True,
                             invalid_raise=False)
    except Exception:
        data = None

    if data is None or data.size == 0 or (data.ndim == 2 and data.shape[1] < 2):
        rows = []
        for ln in lines[skip:]:
            cleaned = ln.replace(delim, " ") if delim else ln
            parts = cleaned.split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
        if not rows:
            raise ValueError(f"Could not parse any numeric rows from {filepath}. "
                             f"First line: {lines[0][:80]!r}")
        data = np.array(rows)

    if data.ndim == 1:
        raise ValueError(f"Only got a 1D array from {filepath}. "
                         f"First line: {lines[0][:80]!r}")

    freqs = data[:, 0]
    mags = data[:, 1]
    valid = np.isfinite(freqs) & np.isfinite(mags)
    return freqs[valid], mags[valid]


def _load_ltspice(lines):
    """
    Parses LTSpice AC-analysis text export of the form:
        Freq.   V(n028)/V(n012)
        1.0e+01 (-2.79e-03dB,-1.13e-01°)
    Returns (frequencies_Hz, magnitudes_dB), discarding the phase column.
    """
    rows = []
    # Skip first line if it's a header (non-numeric first token)
    start = 0
    first_tok = lines[0].split()[0] if lines[0].split() else ""
    try:
        float(first_tok)
    except ValueError:
        start = 1

    for ln in lines[start:]:
        # Split off the frequency from the parenthesised complex value
        # Example: "1.00000000000000e+01\t(-2.79598735535118e-03dB,-1.13723633143227e-01°)"
        parts = ln.split("\t") if "\t" in ln else ln.split(None, 1)
        if len(parts) < 2:
            continue
        freq_str, complex_str = parts[0].strip(), parts[1].strip()

        try:
            freq = float(freq_str)
        except ValueError:
            continue

        # Strip parens, split on comma to separate magnitude and phase
        cs = complex_str.strip("() ")
        if "," not in cs:
            continue
        mag_str = cs.split(",")[0]
        # Remove the trailing 'dB' marker
        mag_str = mag_str.replace("dB", "").strip()
        try:
            mag = float(mag_str)
        except ValueError:
            continue

        rows.append([freq, mag])

    if not rows:
        raise ValueError("LTSpice format detected but no rows parsed.")

    data = np.array(rows)
    return data[:, 0], data[:, 1]

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    traces = []  # list of (freqs, mags, label, style)

    f_ref, m_ref = load_trace(FILE_REF)
    print(f"[INFO] {LABEL_REF}: {len(f_ref)} pts | "
          f"{f_ref.min():.1f} -> {f_ref.max():.1e} Hz | "
          f"{m_ref.min():.2f} -> {m_ref.max():.2f} dB")
    traces.append((f_ref, m_ref, LABEL_REF, {"color": "#d62728", "linestyle": "--", "linewidth": 1.8}))

    f_sim, m_sim = load_trace(FILE_SIM)
    print(f"[INFO] {LABEL_SIM}: {len(f_sim)} pts | "
          f"{f_sim.min():.1f} -> {f_sim.max():.1e} Hz | "
          f"{m_sim.min():.2f} -> {m_sim.max():.2f} dB")
    traces.append((f_sim, m_sim, LABEL_SIM, {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8}))

    if FILE_LTS and os.path.exists(FILE_LTS):
        f_lts, m_lts = load_trace(FILE_LTS)
        print(f"[INFO] {LABEL_LTS}: {len(f_lts)} pts | "
              f"{f_lts.min():.1f} -> {f_lts.max():.1e} Hz | "
              f"{m_lts.min():.2f} -> {m_lts.max():.2f} dB")
        traces.append((f_lts, m_lts, LABEL_LTS, {"color": "#2ca02c", "linestyle": ":", "linewidth": 1.8}))

    fig, ax = plt.subplots(figsize=(10, 5))
    for f, m, label, style in traces:
        ax.plot(f, m, label=label, **style)

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("FRA Trace Comparison")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"[INFO] Plot saved to {OUTPUT_PNG}")
    plt.show()

    import numpy as np
    test_freq = 1e4
    for f, m, label, _ in traces:
        idx = np.argmin(np.abs(f - test_freq))
        print(f"  {label} at {f[idx]:.0f} Hz: {m[idx]:.4f} dB")
