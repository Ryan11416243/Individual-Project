import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
FEATURES_TO_USE = [
    'CC_LF', 'ASLE_LF', 'RMSE_LF',
    'CC_MF', 'ASLE_MF', 'RMSE_MF',
    'CC_HF', 'ASLE_HF', 'RMSE_HF',
    'Peak_1_Hz', 'Peak_2_Hz', 'Peak_3_Hz', 'Peak_4_Hz', 'Peak_5_Hz',
    'Peak_1_dB', 'Peak_2_dB', 'Peak_3_dB', 'Peak_4_dB', 'Peak_5_dB',
]

# Set to 3 classes to match your 85% baseline. 
# Change to [0, 1, 2, 3] if you add a 4th class later.
CLASSES_TO_INCLUDE = [0, 1, 2] 

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading dataset...")
df = pd.read_csv('Master_ML_Dataset.csv')
df.fillna(0, inplace=True)
df = df[df['True_Label'].isin(CLASSES_TO_INCLUDE)]

X = df[FEATURES_TO_USE] 
y = df['True_Label']
feature_names = X.columns

# ==========================================
# 2. STRICT LEAKAGE-FREE DATA SPLITTING
# ==========================================
# A. Create the Raw Split (For Random Forest)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# B. Create the Scaled Split (For SVM and KNN specific training)
scaler = StandardScaler()
# FIT ONLY ON TRAINING DATA to prevent leakage
X_train_scaled = scaler.fit_transform(X_train_raw)
# Transform test data using the training math
X_test_scaled = scaler.transform(X_test_raw)   

# ==========================================
# 3. DEFINE MODELS
# ==========================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=67, class_weight='balanced'),
    "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=67),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

results = []
trained_models = {}
target_names = [f"Class {int(i)}" for i in sorted(y.unique())]

print("Running Strict Leakage-Free Cross-Validation and Training...\n")

# ==========================================
# 4. EVALUATE AND TRAIN
# ==========================================
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
misclassifications = {}

for i, (name, model) in enumerate(models.items()):
    
    if name == "Random Forest":
        # RF gets RAW data directly
        cv_scores = cross_val_score(model, X, y, cv=5)
        model.fit(X_train_raw, y_train)
        y_pred = model.predict(X_test_raw)
    else:
        # SVM and KNN get a PIPELINE for Cross-Validation to prevent leakage between folds
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        
        # Train final model on safely scaled train split, predict on test split
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
    trained_models[name] = model
    results.append({"Model": name, "Mean CV Accuracy": cv_scores.mean() * 100, "Std Dev": cv_scores.std() * 100})
    
    # Store misclassifications for analysis
    cm = confusion_matrix(y_test, y_pred)
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    if cm_copy.sum() > 0:
        actual_idx, pred_idx = np.unravel_index(np.argmax(cm_copy, axis=None), cm_copy.shape)
        misclassifications[name] = (target_names[actual_idx], target_names[pred_idx], cm_copy[actual_idx, pred_idx])
    
    # Plot individual Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i], xticklabels=target_names, yticklabels=target_names)
    axes_cm[i].set_title(f"{name} Matrix")
    axes_cm[i].set_ylabel('Actual')
    axes_cm[i].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('Plot_1_Confusion_Matrices.png', dpi=300)

# ==========================================
# 5. VISUALIZATION: ACCURACY COMPARISON
# ==========================================
comparison_df = pd.DataFrame(results).sort_values(by="Mean CV Accuracy", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Mean CV Accuracy", data=comparison_df, palette="viridis")
plt.errorbar(x=range(len(results)), y=comparison_df["Mean CV Accuracy"], 
             yerr=comparison_df["Std Dev"], fmt='none', c='black', capsize=5)
plt.title("Model Accuracy Comparison (Leakage-Free)")
plt.ylabel("Cross-Validation Accuracy (%)")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('Plot_2_Accuracy_Comparison.png', dpi=300)

# ==========================================
# 6. VISUALIZATION: FEATURE IMPORTANCE
# ==========================================
fig_fi, axes_fi = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Importance
rf_importances = trained_models["Random Forest"].feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]

axes_fi[0].bar(range(X.shape[1]), rf_importances[rf_indices], color='teal')
axes_fi[0].set_xticks(range(X.shape[1]))
axes_fi[0].set_xticklabels([feature_names[idx] for idx in rf_indices], rotation=45, ha='right')
axes_fi[0].set_title("Random Forest: Feature Importance (Gini)")

# SVM Importance (Mean of absolute coefficients)
svm_coefs = np.mean(np.abs(trained_models["SVM (Linear)"].coef_), axis=0)
svm_indices = np.argsort(svm_coefs)[::-1]

axes_fi[1].bar(range(X.shape[1]), svm_coefs[svm_indices], color='darkorange')
axes_fi[1].set_xticks(range(X.shape[1]))
axes_fi[1].set_xticklabels([feature_names[idx] for idx in svm_indices], rotation=45, ha='right')
axes_fi[1].set_title("Linear SVM: Feature Importance (Weights)")

plt.tight_layout()
plt.savefig('Plot_3_Feature_Importance.png', dpi=300)

# ==========================================
# 7. TEXT OUTPUT
# ==========================================
print("\n=== OPTIMIZED MODEL PERFORMANCE SUMMARY ===")
print(comparison_df.to_string(index=False))

print("\n=== MISCLASSIFICATION ANALYSIS ===")
if misclassifications:
    for name, miss in misclassifications.items():
        print(f"{name}: Most frequently confused Actual {miss[0]} as {miss[1]} ({miss[2]} times).")
else:
    print("No prominent misclassifications found!")

print("\nAll plots saved successfully.")
plt.show()