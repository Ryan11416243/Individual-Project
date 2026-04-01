import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# CONFIGURATION & TOGGLES
# ==========================================

# 1. FEATURE TOGGLE: Which columns should the AI look at?
FEATURES_TO_USE = [
    'CC_LF', 'ASLE_LF', 'RMSE_LF',
    'CC_MF', 'ASLE_MF', 'RMSE_MF',
    'CC_HF', 'ASLE_HF', 'RMSE_HF',
    'Peak_1_Hz', 'Peak_2_Hz', 'Peak_3_Hz', 'Peak_4_Hz', 'Peak_5_Hz',
    'Peak_1_dB', 'Peak_2_dB', 'Peak_3_dB', 'Peak_4_dB', 'Peak_5_dB',
]

# 2. CLASS TOGGLE: Which classes do you want to include in this run?
# Example: [0, 1, 2] includes all. [0, 1] would ignore Class 2.
CLASSES_TO_INCLUDE = [0, 1, 2 ,3] 

# ==========================================
# MAIN SCRIPT
# ==========================================

# 1. Load and Clean
df = pd.read_csv('Master_ML_Dataset.csv')
df.fillna(0, inplace=True)

# --- NEW: CLASS FILTERING LOGIC ---
print(f"Original dataset size: {len(df)} samples")
df = df[df['True_Label'].isin(CLASSES_TO_INCLUDE)]
print(f"Filtered dataset size: {len(df)} samples (Classes: {CLASSES_TO_INCLUDE})\n")

# 2. Correlation Heatmap
print("Generating Correlation Heatmap...")
plt.figure(figsize=(12, 8))
corr_matrix = df[FEATURES_TO_USE].corr()
sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0)
plt.title(f'Feature Correlation Matrix (Classes {CLASSES_TO_INCLUDE})')
plt.tight_layout()
plt.savefig('Correlation_Heatmap.png', dpi=300)

# 3. Prepare Data
X = df[FEATURES_TO_USE] 
y = df['True_Label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize and Cross-Validate
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

print("\nPerforming 5-Fold Cross-Validation...")
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"CV Mean Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

# 5. Train Final Model
print("Training final model...")
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# 6. FEATURE IMPORTANCE PLOT
print("Generating Feature Importance Plot...")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title(f"Feature Importance (Classes {CLASSES_TO_INCLUDE})")
plt.bar(range(X.shape[1]), importances[indices], color='teal', align="center")
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Relative Importance Score')
plt.tight_layout()
plt.savefig('Feature_Importance.png', dpi=300)

# 7. Confusion Matrix
# Sorted classes for proper axis labeling
unique_classes = sorted(y.unique())
target_names = [f"Class {int(i)}" for i in unique_classes]
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title(f'Confusion Matrix (Classes: {CLASSES_TO_INCLUDE})')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig('Confusion_Matrix_Final.png', dpi=300)

print("\nProcess Complete. Images saved.")
plt.show()