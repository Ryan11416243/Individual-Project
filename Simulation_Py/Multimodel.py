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
SEVERITIES_TO_INCLUDE = ['0%', 'Incipient', 'Moderate', 'Severe']

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


# DATASET SUBSAMPLING (WIDTH & DEPTH)
# Isolate variables without relying on new physical simulations.
SUBSAMPLE_DATA = True

# WIDTH: How many unique transformers (DNAs) per fault class? (e.g., limit to 100)
MAX_DNAS_PER_CLASS = 150 

# DEPTH: How many historical sweeps per transformer? (e.g., limit to 2)
MAX_SWEEPS_PER_DNA = 3


# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("Loading dataset...")
df = pd.read_csv('Simulation_Py/Master_ML_Dataset.csv')

df['Severity'] = df['Severity'].fillna('0%').astype(str).str.strip()
df.fillna(-1, inplace=True) # Explicitly flag missing peaks as -1

df = df[df['True_Label'].isin(CLASSES_TO_INCLUDE)]
df = df[df['Severity'].isin(SEVERITIES_TO_INCLUDE)]

y = df['True_Label']
groups = df['Unit_ID']

# ---------------------------------------------------------
# STEP 1: FREEZE THE TEST SET FIRST
# We lock away a massive, unseen 20% of the data. 
# This Test Set will NEVER be reduced, ensuring a fair baseline.
# ---------------------------------------------------------
# StratifiedGroupKFold gives us a single fold where:
# - No Unit_ID leaks across train/test
# - Class proportions are approximately preserved in both splits
sgkf_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(sgkf_split.split(df, y, groups=groups))

# Sanity check — print class distribution in both splits
print("[DATA SPLIT] Class distribution in TEST set:")
print(y.iloc[test_idx].value_counts().sort_index())
print("[DATA SPLIT] Class distribution in TRAIN set:")
print(y.iloc[train_idx].value_counts().sort_index())

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print(f"[DATA SPLIT] Massive Test Vault Frozen: {df_test['Unit_ID'].nunique()} Unique Transformers.")

# ---------------------------------------------------------
# STEP 2: STRATIFIED SUBSAMPLING ON TRAINING DATA ONLY
# ---------------------------------------------------------
if SUBSAMPLE_DATA:
    print(f"[DATA PREP] Subsampling Training Set -> Width: {MAX_DNAS_PER_CLASS}/class | Depth: {MAX_SWEEPS_PER_DNA}/DNA")
    filtered_dfs = []
    np.random.seed(42) 

    for label, class_df in df_train.groupby('True_Label'):
        unique_locations = class_df['Location_of_fault'].unique()
        dnas_per_loc = max(1, MAX_DNAS_PER_CLASS // len(unique_locations))
        
        for loc in unique_locations:
            loc_df = class_df[class_df['Location_of_fault'] == loc]
            unique_dnas_in_loc = loc_df['Unit_ID'].unique()
            
            # WIDTH: Extract evenly from each physical location
            if len(unique_dnas_in_loc) > dnas_per_loc:
                selected_dnas = np.random.choice(unique_dnas_in_loc, dnas_per_loc, replace=False)
            else:
                selected_dnas = unique_dnas_in_loc
                
            width_filtered_df = loc_df[loc_df['Unit_ID'].isin(selected_dnas)]
            
            # DEPTH: Randomly sample sweeps instead of sequentially grabbing the top (FRA Fix)
            # lambda to sample safely (in case a unit has fewer sweeps than MAX_SWEEPS)
            depth_filtered_df = width_filtered_df.groupby('Unit_ID').apply(
                lambda x: x.sample(n=min(len(x), MAX_SWEEPS_PER_DNA), random_state=42)
            ).reset_index(drop=True)
            
            filtered_dfs.append(depth_filtered_df)
        
    # Reassemble the Training Set
    df_train = pd.concat(filtered_dfs).reset_index(drop=True)
    y_train = df_train['True_Label']
    groups_train = df_train['Unit_ID']
    print(f"  -> Training Subsampling complete. Train Shape: {df_train.shape} | Test Shape: {df_test.shape}\n")

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


# ==========================================
# 6. EXPERIMENT 3: WIDTH VS. DEPTH
# ==========================================
print("\n=== RUNNING EXPERIMENT 3: WIDTH VS. DEPTH SCALABILITY ===")
# Test how the model responds to more unique units (Width) 
# versus more historical sweeps per unit (Depth).

test_widths = [25, 50, 100, 150, 200]  # Number of unique DNAs per class
test_depths = [1, 2, 3, 4, 5]             # Number of historical sweeps per DNA

# Random Forest for this as it is fast and handles imputed data perfectly
matrix_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

heatmap_data = np.zeros((len(test_depths), len(test_widths)))

df_train_raw = df.iloc[train_idx].copy()   # full, unsubsampled training population
y_train_raw = y.iloc[train_idx]
groups_train_raw = groups.iloc[train_idx]

# Then in Experiment 3, replace df_train with df_train_raw:
for i, depth in enumerate(test_depths):
    for j, width in enumerate(test_widths):
        filtered_dfs = []
        
        for label, class_df in df_train_raw.groupby('True_Label'):  # <-- raw data
            all_dnas = class_df['Unit_ID'].unique()
            
            if len(all_dnas) > width:
                selected_dnas = np.random.choice(all_dnas, width, replace=False)
            else:
                selected_dnas = all_dnas
            
            width_filtered_df = class_df[class_df['Unit_ID'].isin(selected_dnas)]
            
            depth_filtered_df = (
                width_filtered_df
                .groupby('Unit_ID', group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), depth), random_state=42))
                .reset_index(drop=True)
            )
            filtered_dfs.append(depth_filtered_df)
        
        # Apply the exact same Stratified Subsampling logic, but using our loop variables
        for label, class_df in df_train.groupby('True_Label'):
            # Get ALL unique DNAs for this class regardless of location
            all_dnas = class_df['Unit_ID'].unique()
            
            # WIDTH: Sample exactly MAX_DNAS_PER_CLASS units directly,
            # not indirectly via location — this guarantees equal class width
            if len(all_dnas) > MAX_DNAS_PER_CLASS:
                selected_dnas = np.random.choice(all_dnas, MAX_DNAS_PER_CLASS, replace=False)
            else:
                selected_dnas = all_dnas
                print(f"  [WARNING] Class {label} only has {len(all_dnas)} DNAs "
                    f"(less than MAX_DNAS_PER_CLASS={MAX_DNAS_PER_CLASS})")
            
            width_filtered_df = class_df[class_df['Unit_ID'].isin(selected_dnas)]
            
            # DEPTH: Sample randomly per DNA, not sequentially
            depth_filtered_df = (
                width_filtered_df
                .groupby('Unit_ID', group_keys=False)
                .apply(lambda x: x.sample(n=min(len(x), MAX_SWEEPS_PER_DNA), random_state=42))
                .reset_index(drop=True)
            )
            
            filtered_dfs.append(depth_filtered_df)

        # Verify the result is balanced
        df_train = pd.concat(filtered_dfs).reset_index(drop=True)
        print("\n[SUBSAMPLING] Final training samples per class:")
        print(df_train['True_Label'].value_counts().sort_index())
        
        # Assemble the specific training subset for this Grid coordinate
        df_subset = pd.concat(filtered_dfs).reset_index(drop=True)
        X_train_sub = df_subset[ALL_FEATURES]
        y_train_sub = df_subset['True_Label']
        
        # Train and Evaluate on the Frozen Vault
        matrix_clf.fit(X_train_sub, y_train_sub)
        acc = matrix_clf.score(X_test_full, y_test) * 100
        
        heatmap_data[i, j] = acc
        print(f"  -> Width: {width} DNAs/Class | Depth: {depth} Sweeps/DNA | Acc: {acc:.2f}%")

# Plotting the 2D Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", 
            xticklabels=test_widths, yticklabels=test_depths, cbar_kws={'label': 'Test Accuracy (%)'})

plt.title("Data Scalability Matrix: Fleet Width vs. Historical Depth")
plt.xlabel("Width: Unique Transformers (DNAs) per Class")
plt.ylabel("Depth: Historical Sweeps per Transformer")
plt.tight_layout()
plt.savefig('Plot_3_Width_vs_Depth.png', dpi=300)