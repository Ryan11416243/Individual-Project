import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, RandomizedSearchCV

# ==========================================
# 1. CONFIGURATION & FEATURE SETS
# ==========================================
CLASSES_TO_INCLUDE = [0, 1, 2, 3, 4] 
SEVERITIES_TO_INCLUDE = ['0%', '5%', '10%', '20%']

ALL_FEATURES = [
    'CCF_LF', 'LCC_LF', 'SDA_LF', 'SE_LF', 'CSD_LF',
    'CCF_MF', 'LCC_MF', 'SDA_MF', 'SE_MF', 'CSD_MF',
    'CCF_HF', 'LCC_HF', 'SDA_HF', 'SE_HF', 'CSD_HF',
    'Peak_1_Hz', 'Peak_2_Hz', 'Peak_3_Hz', 'Peak_4_Hz', 'Peak_5_Hz',
    'Peak_1_dB', 'Peak_2_dB', 'Peak_3_dB', 'Peak_4_dB', 'Peak_5_dB',
]

# FRA EXPERIMENT: We define specific physical subsets to test
FEATURE_EXPERIMENTS = {
    "All Features": ALL_FEATURES,
    "High-Freq & Peaks (Capacitive)": [f for f in ALL_FEATURES if 'HF' in f or 'Peak' in f],
    "Low & Mid-Freq (Core/Inductive)": [f for f in ALL_FEATURES if 'LF' in f or 'MF' in f],
    "Statistical Indices Only": [f for f in ALL_FEATURES if 'Peak' not in f],
    "Resonant Peaks Only": [f for f in ALL_FEATURES if 'Peak' in f]
}

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("Loading dataset...")
df = pd.read_csv('Simulation_Py/Master_ML_Dataset.csv')

df['Severity'] = df['Severity'].fillna('0%').astype(str).str.strip()
df.fillna(-1, inplace=True) # CS Fix: Explicitly flag missing peaks as -1

df = df[df['True_Label'].isin(CLASSES_TO_INCLUDE)]
df = df[df['Severity'].isin(SEVERITIES_TO_INCLUDE)]

y = df['True_Label']
groups = df['Unit_ID']

# Safe Group Split for Final Testing
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, groups=groups))

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# Cross-Validation Strategy for Tuning
# ==========================================
# DYNAMIC CROSS-VALIDATION SIZING
# ==========================================
# Count how many unique transformers (groups) actually ended up in the training set
n_unique_groups = groups_train.nunique()
safe_splits = min(5, n_unique_groups)

if safe_splits < 2:
    raise ValueError(f"FATAL: Only {n_unique_groups} unique Unit_IDs found. Need at least 2.")

# CS FIX: StratifiedGroupKFold ensures no fold accidentally ends up with only 1 class
sgkf = StratifiedGroupKFold(n_splits=safe_splits)

# ==========================================
# 3. EXPERIMENT 1: FEATURE ABLATION (FRA FOCUS)
# ==========================================
print("\n=== RUNNING FRA FEATURE ABLATION STUDY ===")
ablation_results = []

# We use a fast, untuned Random Forest just to test the feature sets
base_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

for exp_name, features in FEATURE_EXPERIMENTS.items():
    X_train_exp = df_train[features]
    
    # Calculate Cross-Validation Score for this specific feature set
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(base_rf, X_train_exp, y_train, cv=sgkf, groups=groups_train)
    
    mean_acc = scores.mean() * 100
    ablation_results.append({"Feature Set": exp_name, "Accuracy": mean_acc})
    print(f"  -> {exp_name}: {mean_acc:.2f}%")

# Plotting Feature Ablation
ablation_df = pd.DataFrame(ablation_results).sort_values(by="Accuracy", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Feature Set", data=ablation_df, palette="magma")
plt.title("FRA Feature Ablation: Where is the Diagnostic Signal?")
plt.xlabel("Cross-Validation Accuracy (%)")
plt.tight_layout()
plt.savefig('Plot_1_Feature_Ablation.png', dpi=300)

# ==========================================
# 4. EXPERIMENT 2: HYPERPARAMETER TUNING (CS FOCUS)
# ==========================================
print("\n=== RUNNING CS HYPERPARAMETER TUNING ===")
# We will use ALL FEATURES for the final tuned models
X_train_full = df_train[ALL_FEATURES]
X_test_full = df_test[ALL_FEATURES]

# Define Pipelines (Scaling is required for SVM)
pipelines = {
    "Random Forest": Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ]),
    "Gradient Boosting": Pipeline([
        ('classifier', GradientBoostingClassifier(random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42))
    ])
}

# Define the Grid Search spaces for each algorithm
param_grids = {
    "Random Forest": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    },
    "SVM (RBF)": {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__kernel': ['rbf', 'linear']
    }
}

best_models = {}
tuning_results = []

for name in pipelines.keys():
    print(f"Tuning {name}...")
    
    # CS FIX: RandomizedSearchCV is faster than GridSearch and prevents overfitting.
    # We pass `gkf` to ensure cross-validation respects our Unit_ID groups!
    search = RandomizedSearchCV(
        pipelines[name], 
        param_grids[name], 
        n_iter=10,          # Tests 10 random combinations from the grid
        cv=sgkf,             # Group-aware cross validation
        scoring='accuracy', 
        random_state=42,
        n_jobs=-1           # Use all CPU cores
    )
    
    # Fit the search (Passing groups_train is strictly required for gkf)
    search.fit(X_train_full, y_train, groups=groups_train)
    
    best_models[name] = search.best_estimator_
    
    # Evaluate on the completely unseen TEST set
    test_accuracy = search.score(X_test_full, y_test) * 100
    
    tuning_results.append({
        "Model": name, 
        "Best CV Score": search.best_score_ * 100,
        "Unseen Test Accuracy": test_accuracy,
        "Best Params": str(search.best_params_)
    })
    print(f"  -> Best CV: {search.best_score_ * 100:.2f}% | Test Set: {test_accuracy:.2f}%")

# Plotting Tuned Model Performance
tuned_df = pd.DataFrame(tuning_results)
plt.figure(figsize=(8, 5))
# Melt dataframe for easy grouped bar plotting
melted_df = pd.melt(tuned_df, id_vars=['Model'], value_vars=['Best CV Score', 'Unseen Test Accuracy'], 
                    var_name='Metric', value_name='Accuracy (%)')

sns.barplot(x='Model', y='Accuracy (%)', hue='Metric', data=melted_df, palette="crest")
plt.title("Optimized Model Performance (Group Leakage-Free)")
plt.ylim(0, 105)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('Plot_2_Tuned_Performance.png', dpi=300)

# ==========================================
# 5. SUMMARY OUPUT
# ==========================================
print("\n=== HYPERPARAMETER OPTIMIZATION SUMMARY ===")
for res in tuning_results:
    print(f"\nModel: {res['Model']}")
    print(f"Test Accuracy: {res['Unseen Test Accuracy']:.2f}%")
    print(f"Optimal Parameters Found: {res['Best Params']}")

print("\nAll experiments complete. Plots saved to disk.")