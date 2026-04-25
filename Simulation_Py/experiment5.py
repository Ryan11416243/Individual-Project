import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

# 1. Import your existing logic so we don't repeat code!
from extraction_sim import extract_features_single, read_sim_data, CLASS_LABELS
from Multimodel import ALL_FEATURES

# ==========================================
# 1. FAST-TRAIN THE CHAMPION MODEL
# ==========================================
print("Loading Master Dataset for training...")
df = pd.read_csv('Simulation_Py/Master_ML_Dataset.csv')

# Train on the clean baseline data
X_train = df[ALL_FEATURES]
y_train = df['True_Label']

champion_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('classifier', GradientBoostingClassifier(n_estimators=50, max_depth=7, learning_rate=0.2, random_state=42))
])

print("Training Gradient Boosting Champion...")
champion_pipeline.fit(X_train, y_train)
print("Model trained. Ready for Physical Sensitivity Sweep.")

# ==========================================
# 2. FREQUENCY-DEPENDENT NOISE INJECTOR
# ==========================================
def inject_hardware_noise(magnitudes_dB, frequencies, noise_level):
    if noise_level == 0.0:
        return magnitudes_dB
    # Base flat noise
    noise = np.random.normal(0, noise_level, len(magnitudes_dB))
    # High-frequency penalty scalar
    freq_weight = (frequencies / np.max(frequencies)) ** 1.5
    noise += np.random.normal(0, noise_level * freq_weight)
    return magnitudes_dB + noise

# ==========================================
# 3. AUTOMATED SENSITIVITY SWEEP
# ==========================================
DATASET_DIR = "Simulation_Py/Dataset"  # Make sure this points to your raw text files
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
MAX_FILES_PER_CLASS = 20 # Keep it small for fast testing (e.g., 20 files per fault type)

sensitivity_results = []

for noise in NOISE_LEVELS:
    print(f"\nEvaluating Physical Noise Floor: {noise*100:.1f}%...")
    
    y_true = []
    y_pred = []
    
    # Crawl the dataset folder
    for folder_name in os.listdir(DATASET_DIR):
        target_folder = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(target_folder): continue
        
        # Parse label
        fault_type = "Healthy" if folder_name == "Simulation_Healthy" else folder_name.split('_')[1]
        label = CLASS_LABELS.get(fault_type, -1)
        if label == -1: continue
            
        unit_folders = glob.glob(os.path.join(target_folder, 'Unit_*'))[:MAX_FILES_PER_CLASS]
        
        for unit_path in unit_folders:
            ref_path = os.path.join(unit_path, 'Trace_0.txt')
            if not os.path.exists(ref_path): continue
                
            ref_freqs, ref_mags = read_sim_data(ref_path)
            
            trace_files = [f for f in glob.glob(os.path.join(unit_path, '*.txt')) if 'Trace_0' not in f]
            if not trace_files: continue
                
            # Just grab one trace per unit for speed during the test
            test_trace = trace_files[0]
            freqs, mags = read_sim_data(test_trace)
            
            # THE MAGIC: Inject physical noise into the raw array BEFORE extraction
            noisy_mags = inject_hardware_noise(mags, freqs, noise)
            
            # Extract features from the degraded signal
            features = extract_features_single(freqs, noisy_mags, ref_mags, "Test", "Unit")
            
            # Format for prediction
            feature_df = pd.DataFrame([features])[ALL_FEATURES]
            prediction = champion_pipeline.predict(feature_df)[0]
            
            y_true.append(label)
            y_pred.append(prediction)
            
    # Calculate accuracy for this noise level
    acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
    sensitivity_results.append({"Noise Level (%)": noise * 100, "Accuracy": acc})
    print(f"  -> Accuracy at {noise*100:.1f}% Noise: {acc:.2f}%")

# ==========================================
# 4. GENERATE THE FINAL PLOT
# ==========================================
sens_df = pd.DataFrame(sensitivity_results)
plt.figure(figsize=(8, 5))
sns.lineplot(x="Noise Level (%)", y="Accuracy", data=sens_df, marker='o', color='#d62728', linewidth=2.5, markersize=8)

plt.title("Algorithmic Resilience to Real-World Hardware Degradation")
plt.xlabel("Physical Measurement Noise Floor (%)")
plt.ylabel("Diagnostic Accuracy (%)")
plt.ylim(0, 105)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Plot_5_Physical_Sensitivity.png', dpi=300)
print("\nSensitivity Sweep Complete. Plot saved.")