import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ==========================================
# 1. UNIVERSAL FILE PARSER
# ==========================================
def read_fra_file(filepath):
    """
    Intelligently reads either LTSpice or Python generated FRA text files.
    - LTSpice: 1.000e+01 \t (-2.795e-03dB,-1.137e-01°)
    - Python:  1.000e+01 \t -2.795e-03
    """
    freqs = []
    mags = []
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find file '{filepath}'")
        return None, None
        
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            # Skip empty lines or headers
            if not line.strip() or line.lower().startswith('freq') or line.lower().startswith('step'):
                continue
            
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # The first column is always frequency
                    freq = float(parts[0])
                    
                    # Detect LTSpice format by looking for parentheses and 'dB'
                    if '(' in parts[1] and 'dB' in parts[1]:
                        match = re.search(r'\(([^d]+)dB', parts[1])
                        if match:
                            mag = float(match.group(1))
                            freqs.append(freq)
                            mags.append(mag)
                    else:
                        # Assume Python format (just a raw number for magnitude)
                        mag = float(parts[1])
                        freqs.append(freq)
                        mags.append(mag)
                except ValueError:
                    continue # Ignore lines that fail to parse
                    
    return np.array(freqs), np.array(mags)

# ==========================================
# 2. STATISTICAL EVALUATION ENGINE
# ==========================================
def calculate_metrics(f1, m1, f2, m2):
    """
    Calculates RMSE, Max Absolute Error, and CCF.
    Uses interpolation to handle misaligned frequency steps between simulators.
    """
    # 1. Find the overlapping frequency range
    min_f = max(f1[0], f2[0])
    max_f = min(f1[-1], f2[-1])
    
    # 2. Filter dataset 1 to only include the overlapping range
    mask = (f1 >= min_f) & (f1 <= max_f)
    f_common = f1[mask]
    m1_common = m1[mask]
    
    # 3. Interpolate dataset 2 onto dataset 1's frequency steps
    m2_interp = np.interp(f_common, f2, m2)
    
    # 4. Calculate Math Metrics
    rmse = np.sqrt(np.mean((m1_common - m2_interp)**2))
    max_error = np.max(np.abs(m1_common - m2_interp))
    
    # CCF Calculation
    mean1 = np.mean(m1_common)
    mean2 = np.mean(m2_interp)
    numerator = np.sum((m1_common - mean1) * (m2_interp - mean2))
    denominator = np.sqrt(np.sum((m1_common - mean1)**2) * np.sum((m2_interp - mean2)**2))
    
    # Prevent divide by zero error if comparing a flat line
    ccf = numerator / denominator if denominator != 0 else 0
    
    return rmse, max_error, ccf

# ==========================================
# 3. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Put exact filenames here.
    file_1_path = "Simulation_Py\sim_axial_5.txt"   
    file_2_path = "Simulation_Py\ltspice_axial.txt"           
    
    label_1 = "Python Model"
    label_2 = "LTSpice Export"
    # ---------------------

    print(f"Loading {file_1_path}...")
    f1, m1 = read_fra_file(file_1_path)
    
    print(f"Loading {file_2_path}...")
    f2, m2 = read_fra_file(file_2_path)
    
    if f1 is not None and f2 is not None:
        # Calculate Metrics
        rmse, max_err, ccf = calculate_metrics(f1, m1, f2, m2)
        
        print("\n=== Statistical Comparison ===")
        print(f"RMSE:               {rmse:.4f} dB")
        print(f"Max Absolute Error: {max_err:.4f} dB")
        print(f"CCF:                {ccf:.6f}")
        print("==============================\n")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Plot File 1 as a solid blue line
        plt.semilogx(f1, m1, color='blue', linewidth=2, label=label_1)
        
        # Plot File 2 as a dashed red line to easily see overlaps
        plt.semilogx(f2, m2, color='red', linestyle='--', linewidth=2, label=label_2)
        
        plt.title(f"FRA Comparison: {label_1} vs {label_2}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Add a text box inside the plot with the stats
        stats_text = f"RMSE: {rmse:.4f} dB\nMax Err: {max_err:.4f} dB\nCCF: {ccf:.6f}"
        plt.gca().text(0.02, 0.05, stats_text, transform=plt.gca().transAxes, 
                       fontsize=10, verticalalignment='bottom', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend(loc="lower right")
        plt.show()