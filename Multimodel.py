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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler



# ==========================================
# 1. CONFIGURATION & FEATURE SETS
# ==========================================
CLASSES_TO_INCLUDE = [0, 1, 2, 3, 4] 
SEVERITIES_TO_INCLUDE = ['0%', 'Incipient', 'Moderate', 'Severe']

ALL_FEATURES = [
    'CCF_LF', 'LCC_LF', 'SDA_LF', 'SE_LF', 'CSD_LF',
    'CCF_MF', 'LCC_MF', 'SDA_MF', 'SE_MF', 'CSD_MF',
    'CCF_HF', 'LCC_HF', 'SDA_HF', 'SE_HF', 'CSD_HF',
    'SDA_Win1', 'CCF_Win1', 'CSD_Win1', 'Delta_Peak_1_Hz', 'Delta_Peak_1_dB', 'Delta_Energy_Win1',
    'SDA_Win2', 'CCF_Win2', 'CSD_Win2', 'Delta_Peak_2_Hz', 'Delta_Peak_2_dB', 'Delta_Energy_Win2',
    'SDA_Win3', 'CCF_Win3', 'CSD_Win3', 'Delta_Peak_3_Hz', 'Delta_Peak_3_dB', 'Delta_Energy_Win3',
    'SDA_Win4', 'CCF_Win4', 'CSD_Win4', 'Delta_Peak_4_Hz', 'Delta_Peak_4_dB', 'Delta_Energy_Win4',
    'SDA_Win5', 'CCF_Win5', 'CSD_Win5', 'Delta_Peak_5_Hz', 'Delta_Peak_5_dB', 'Delta_Energy_Win5',
]
# FRA EXPERIMENT: We define specific physical subsets to test
HV_PHYSICAL_SETS = {
    # 1. First anti-resonance region (core + winding capacitances)
    "HV Physical: First Anti-Resonance (LF + Win1)": [
        'CCF_LF', 'LCC_LF', 'SDA_LF', 'SE_LF', 'CSD_LF',          # core region
        'SDA_Win1', 'CCF_Win1', 'CSD_Win1',                       # window around 10 Hz–10 kHz
        'Delta_Peak_1_Hz', 'Delta_Peak_1_dB', 'Delta_Energy_Win1' # peak shift at anti-resonance
    ],
    
    # 2. Smooth capacitive rising tail (HF region, Win4 & Win5)
    "HV Physical: Capacitive High-Freq Tail": [
        'CCF_HF', 'SDA_HF', 'SE_HF', 'CSD_HF',                    # HF statistical indices
        'SDA_Win4', 'CCF_Win4', 'CSD_Win4',                       # 300–600 kHz
        'Delta_Peak_4_Hz', 'Delta_Peak_4_dB', 'Delta_Energy_Win4',
        'SDA_Win5', 'CCF_Win5', 'CSD_Win5',                       # 600 kHz–1 MHz (steep slope)
        'Delta_Peak_5_Hz', 'Delta_Peak_5_dB', 'Delta_Energy_Win5'
    ],
    
    # 3. Winding interaction region (MF + Win2) – where HV deformation affects LV FRA
    "HV Physical: Winding Interaction (MF + Win2)": [
        'CCF_MF', 'LCC_MF', 'SDA_MF', 'SE_MF', 'CSD_MF',          # 2–20 kHz
        'SDA_Win2', 'CCF_Win2', 'CSD_Win2',                       # 10–100 kHz
        'Delta_Peak_2_Hz', 'Delta_Peak_2_dB', 'Delta_Energy_Win2'
    ],
    
    # 4. Combined HV-optimised: exclude very high HF noise (>300 kHz)
    "HV Optimised: Exclude >300 kHz (Noise Reduction)": [
        feat for feat in ALL_FEATURES 
        if not any(x in feat for x in ['Win4', 'Win5', '_HF'])  # keep only LF, MF, Win1-3
    ]
}

# Merge with existing experiments (original sets are kept)
FEATURE_EXPERIMENTS = {
    "All Features": ALL_FEATURES,
    "HV Physical: First Anti-Resonance (LF + Win1)": [
        'CCF_LF', 'LCC_LF', 'SDA_LF', 'SE_LF', 'CSD_LF',          # core region
        'SDA_Win1', 'CCF_Win1', 'CSD_Win1',                       # window around 10 Hz–10 kHz
        'Delta_Peak_1_Hz', 'Delta_Peak_1_dB', 'Delta_Energy_Win1' # peak shift at anti-resonance
    ],
    
    # 2. Smooth capacitive rising tail (HF region, Win4 & Win5)
    "HV Physical: Capacitive High-Freq Tail": [
        'CCF_HF', 'SDA_HF', 'SE_HF', 'CSD_HF',                    # HF statistical indices
        'SDA_Win4', 'CCF_Win4', 'CSD_Win4',                       # 300–600 kHz
        'Delta_Peak_4_Hz', 'Delta_Peak_4_dB', 'Delta_Energy_Win4',
        'SDA_Win5', 'CCF_Win5', 'CSD_Win5',                       # 600 kHz–1 MHz (steep slope)
        'Delta_Peak_5_Hz', 'Delta_Peak_5_dB', 'Delta_Energy_Win5'
    ],
    
    # 3. Winding interaction region (MF + Win2) – where HV deformation affects LV FRA
    "HV Physical: Winding Interaction (MF + Win2)": [
        'CCF_MF', 'LCC_MF', 'SDA_MF', 'SE_MF', 'CSD_MF',          # 2–20 kHz
        'SDA_Win2', 'CCF_Win2', 'CSD_Win2',                       # 10–100 kHz
        'Delta_Peak_2_Hz', 'Delta_Peak_2_dB', 'Delta_Energy_Win2'
    ],
    
    # 4. Combined HV-optimised: exclude very high HF noise (>300 kHz)
    "HV Optimised: Exclude >300 kHz (Noise Reduction)": [
        feat for feat in ALL_FEATURES 
        if not any(x in feat for x in ['Win4', 'Win5', '_HF'])  # keep only LF, MF, Win1-3
    ],
}


# DATASET SUBSAMPLING (WIDTH & DEPTH)
# Isolate variables without relying on new physical simulations.
SUBSAMPLE_DATA = True

# WIDTH: How many unique transformers (DNAs) per fault class? (e.g., limit to 100)
MAX_DNAS_PER_CLASS = 200 

# DEPTH: How many historical sweeps per transformer? (e.g., limit to 2)
MAX_SWEEPS_PER_DNA = 7


# ==========================================
# 2. DATA PREPARATION
# ==========================================
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv('Master_ML_Dataset_HV.csv')

    df['Location_of_fault'] = df['Location_of_fault'].fillna('Global').astype(str)

    df['Severity'] = df['Severity'].fillna('0%').astype(str).str.strip()




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
                depth_filtered_df = (
                    width_filtered_df
                    .groupby('Unit_ID', group_keys=False)
                    .apply(lambda x: x.sample(n=min(len(x), MAX_SWEEPS_PER_DNA), random_state=42))
                    .reset_index(drop=True)
                )
                
                filtered_dfs.append(depth_filtered_df)
            
        # Reassemble the Training Set
        df_train = pd.concat(filtered_dfs).reset_index(drop=True)
        y_train = df_train['True_Label']
        groups_train = df_train['Unit_ID']
        print(f"  -> Training Subsampling complete. Train Shape: {df_train.shape} | Test Shape: {df_test.shape}\n")
        print("\n[SUBSAMPLING] Unique DNAs per class in training set:")
        print(df_train.groupby('True_Label')['Unit_ID'].nunique())

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
    sns.barplot(x="Accuracy", y="Feature Set", data=ablation_df, palette="magma", legend=False)
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
    # imputer for  
    pipelines = {
        "Random Forest": Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        "Gradient Boosting": Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
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
            'classifier__gamma': ['scale', 0.01, 0.1],
            'classifier__kernel': ['rbf'] 
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



    # ============================================
    # Plot 4 Confusion Matrix
    # ============================================


    best_model = best_models["Gradient Boosting"]
    y_pred = best_model.predict(X_test_full)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(5)])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Gradient Boosting — Confusion Matrix (Unseen Test Set)")
    plt.tight_layout()
    plt.savefig('Plot_4_Confusion_Matrix.png', dpi=300)
    print("\n=== PER-CLASS CLASSIFICATION REPORT (Gradient Boosting) ===")
    print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(5)]))

    # ============================================
    # Plot 5
    # ============================================

    cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, 
                                        display_labels=[f"Class {i}" for i in range(5)])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp_norm.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Gradient Boosting — Normalised Confusion Matrix (Recall)")
    plt.tight_layout()
    plt.savefig('Plot_4b_Confusion_Normalised.png', dpi=300)

    misclassified_mask = y_pred != y_test.values
    misclassified_df = df_test.copy()
    misclassified_df['Predicted'] = y_pred
    misclassified_df = misclassified_df[misclassified_mask]

    print("\n=== MISCLASSIFICATION SEVERITY BREAKDOWN ===")
    print(misclassified_df[misclassified_df['True_Label'].isin([0,1,2])]
        .groupby(['True_Label', 'Severity'])
        .size()
        .rename('Count'))

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
                    .apply(lambda x: x.sample(n=min(len(x), depth), random_state=42), include_groups=False)
                    .reset_index(drop=True)
                )
                filtered_dfs.append(depth_filtered_df)
            
            # Apply the exact same Stratified Subsampling logic, but using our loop variables
            
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


    # ==========================================
    # 7. EXPERIMENT 4: SEVERITY THRESHOLD ANALYSIS
    # ==========================================
    print("\n=== RUNNING EXPERIMENT 4: MODERATE/SEVERE ISOLATION ===")
    # How well does the model perform if we ignore "Incipient" faults 
    # that naturally overlap with healthy manufacturing variance?

    # Filter the test set to only include Healthy (0%), Moderate, and Severe
    # We drop 'Incipient' from the test evaluation
    mask_clear_faults = df_test['Severity'].isin(['0%', 'Moderate', 'Severe'])
    X_test_clear = df_test[mask_clear_faults][ALL_FEATURES]
    y_test_clear = df_test[mask_clear_faults]['True_Label']

    best_gb = best_models["Gradient Boosting"]
    clear_acc = best_gb.score(X_test_clear, y_test_clear) * 100

    print(f"  -> Accuracy on Full Test Set: {best_gb.score(X_test_full, y_test) * 100:.2f}%")
    print(f"  -> Accuracy excluding 'Incipient' overlaps: {clear_acc:.2f}%")


    # ==========================================
    # 8. PREDICTIVE CONFIDENCE ANALYSIS
    # ==========================================
    print("\n=== PREDICTIVE CONFIDENCE ANALYSIS (Gradient Boosting) ===")
    # Get the probability scores for every prediction
    y_proba = best_models["Gradient Boosting"].predict_proba(X_test_full)

    # Find the maximum confidence score for each prediction
    confidence_scores = np.max(y_proba, axis=1)

    y_pred_gb = best_models["Gradient Boosting"].predict(X_test_full)
    correct_mask = (y_pred_gb == y_test.values)
    wrong_mask = (y_pred_gb != y_test.values)

    print(f"  -> Average Confidence on CORRECT predictions: {np.mean(confidence_scores[correct_mask]) * 100:.2f}%")
    if np.sum(wrong_mask) > 0:
        print(f"  -> Average Confidence on INCORRECT predictions: {np.mean(confidence_scores[wrong_mask]) * 100:.2f}%")




    # ==========================================
    # PLOT 6: PREDICTIVE CONFIDENCE DISTRIBUTION
    # ==========================================
    plt.figure(figsize=(8, 5))
    sns.kdeplot(confidence_scores[correct_mask] * 100, fill=True, label="Correct Predictions", color="#2ca02c", alpha=0.6)
    if np.sum(wrong_mask) > 0:
        sns.kdeplot(confidence_scores[wrong_mask] * 100, fill=True, label="Incorrect Predictions", color="#d62728", alpha=0.6)

    plt.title("Predictive Uncertainty: Correct vs. Incorrect Diagnoses")
    plt.xlabel("Algorithm Confidence Score (%)")
    plt.ylabel("Density")
    plt.xlim(40, 105)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Plot_6_Confidence_Distribution.png', dpi=300)


    # ==========================================
    # PLOT 7: ACCURACY BY SEVERITY TIER
    # ==========================================
    sev_acc = []
    for sev in ['0%', 'Incipient', 'Moderate', 'Severe']:
        mask = df_test['Severity'] == sev
        if mask.sum() > 0: # Ensure data exists
            acc = best_models["Gradient Boosting"].score(df_test[mask][ALL_FEATURES], df_test[mask]['True_Label'])
            sev_acc.append({"Severity": sev if sev != '0%' else 'Healthy Baseline', "Accuracy": acc * 100})

    sev_df = pd.DataFrame(sev_acc)

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Severity", y="Accuracy", data=sev_df, palette="mako")
    plt.title("Diagnostic Accuracy Scaling by Physical Deformation Severity")
    plt.ylim(0, 110)
    plt.ylabel("Test Set Accuracy (%)")

    # Add percentage labels on top of the bars
    for index, row in sev_df.iterrows():
        ax.text(index, row.Accuracy + 1.5, f'{row.Accuracy:.2f}%', color='black', ha="center", weight='bold')

    plt.tight_layout()
    plt.savefig('Plot_7_Accuracy_by_Severity.png', dpi=300)


    # ==========================================
    # PLOT 8: TOP 15 FEATURE IMPORTANCES
    # ==========================================
    # Extract the internal feature importances from the tuned Gradient Boosting model
    importances = best_models["Gradient Boosting"].named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Grab the top 15

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices], y=np.array(ALL_FEATURES)[indices], palette="rocket")
    plt.title("Top 15 Most Diagnostic FRA Features (Gradient Boosting)")
    plt.xlabel("Relative Importance (Gini/Gain)")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig('Plot_8_Feature_Importance.png', dpi=300)


    # ==========================================
    # PLOT 9: SPATIAL SENSITIVITY (ACCURACY BY LOCATION)
    # ==========================================
    print("\n=== RUNNING EXPERIMENT 5: SPATIAL SENSITIVITY ===")
    loc_acc = []
    localized_test_df = df_test[df_test['Location_of_fault'] != 'Global']
    
    for loc in localized_test_df['Location_of_fault'].unique():
        mask = localized_test_df['Location_of_fault'] == loc
        if mask.sum() > 0:
            acc = best_models["Gradient Boosting"].score(
                localized_test_df[mask][ALL_FEATURES], 
                localized_test_df[mask]['True_Label']
            )
            loc_acc.append({"Location": loc, "Accuracy": acc * 100})

    loc_df = pd.DataFrame(loc_acc).sort_values(by="Location").reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x="Location", y="Accuracy", data=loc_df, palette="viridis")
    plt.title("Spatial Sensitivity: Diagnostic Accuracy by Physical Fault Location")
    plt.ylim(0, 110)
    plt.ylabel("Test Set Accuracy (%)")
    plt.xticks(rotation=45)
    
    for index, row in loc_df.iterrows():
        ax.text(index, row.Accuracy + 1.5, f'{row.Accuracy:.1f}%', color='black', ha="center", weight='bold')

    plt.tight_layout()
    plt.savefig('Plot_9_Accuracy_by_Location.png', dpi=300)

    # ==========================================
    # PLOT 10: MULTI-MODEL CLASS RECALL COMPARISON
    # ==========================================
    print("\n=== RUNNING EXPERIMENT 6: MULTI-MODEL ARCHITECTURE COMPARISON ===")
    CLASS_LABELS_REVERSE = {0: 'Healthy', 1: 'Radial', 2: 'Axial', 3: 'LCP', 4: 'TTSC'}
    comparison_data = []
    
    for model_name, model_pipeline in best_models.items():
        y_pred_multi = model_pipeline.predict(X_test_full)
        report = classification_report(y_test, y_pred_multi, output_dict=True)
        
        for i in range(5):
            class_key = str(i)
            comparison_data.append({
                "Algorithm": model_name,
                "Fault Class": CLASS_LABELS_REVERSE.get(i, f"Class {i}"), 
                "Recall (%)": report[class_key]['recall'] * 100
            })
            
    compare_df = pd.DataFrame(comparison_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Fault Class", y="Recall (%)", hue="Algorithm", data=compare_df, palette="rocket")
    plt.title("Algorithm Architecture Comparison: Recall by Fault Geometry")
    plt.ylim(0, 110)
    plt.ylabel("Sensitivity / Recall (%)")
    plt.legend(title="Algorithm", loc='lower right')
    plt.axhline(90, color='gray', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('Plot_10_MultiModel_Comparison.png', dpi=300)
    print("  -> Experiments complete. All plots saved.")


    # ==========================================
    # PLOT 11: HIGH-DIMENSIONAL FEATURE MANIFOLD (t-SNE)
    # ==========================================
    print("\n=== RUNNING EXPERIMENT 7: t-SNE FEATURE SPACE VISUALIZATION ===")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import time
    
    # We use the test set to show how UNSEEN data clusters
    # Drop NaNs just in case using the Imputer
    imputer = KNNImputer(n_neighbors=5)
    X_test_imputed = imputer.fit_transform(X_test_full)
    
    # t-SNE requires strictly scaled data to calculate distances correctly
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_imputed)
    
    print("  -> Calculating t-SNE embeddings (This may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_test_scaled)
    
    tsne_df = pd.DataFrame({
        'TSNE_1': X_tsne[:, 0],
        'TSNE_2': X_tsne[:, 1],
        'Fault Class': df_test['True_Label'].map(CLASS_LABELS_REVERSE),
        'Severity': df_test['Severity']
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='TSNE_1', y='TSNE_2',
        hue='Fault Class',
        style='Severity',
        palette='bright',
        data=tsne_df,
        alpha=0.7,
        s=40
    )
    plt.title("t-SNE Projection of the 45-Dimensional FRA Feature Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Plot_11_tSNE_Clusters.png', dpi=300)
    print("  -> t-SNE Manifold saved as Plot_11.")


    # ==========================================
    # PLOT 13: STATISTICAL STABILITY (CV VARIANCE)
    # ==========================================
    print("\n=== RUNNING EXPERIMENT 9: CROSS-VALIDATION STABILITY ===")
    
    # We will re-run a quick standard cross_val_score on the best models 
    # to extract the raw fold arrays cleanly.
    from sklearn.model_selection import cross_val_score
    
    cv_variance_data = []
    
    for model_name, model_pipeline in best_models.items():
        print(f"  -> Extracting CV folds for {model_name}...")
        # Use the same sgkf to ensure no data leakage across folds
        fold_scores = cross_val_score(
            model_pipeline, 
            X_train_full, 
            y_train, 
            cv=sgkf, 
            groups=groups_train, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        for fold_idx, score in enumerate(fold_scores):
            cv_variance_data.append({
                "Algorithm": model_name,
                "Fold": f"Fold {fold_idx + 1}",
                "Validation Accuracy (%)": score * 100
            })
            
    cv_df = pd.DataFrame(cv_variance_data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Algorithm", y="Validation Accuracy (%)", data=cv_df, palette="crest", width=0.5)
    sns.stripplot(x="Algorithm", y="Validation Accuracy (%)", data=cv_df, color=".3", linewidth=1, alpha=0.7)
    
    plt.title("Statistical Stability: Variance Across K-Fold Cross Validation")
    plt.ylabel("Validation Accuracy (%)")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    plt.savefig('Plot_13_CV_Stability.png', dpi=300)
    print("  -> CV Stability Boxplots saved as Plot_13.")


    # ==========================================
    # PLOT 8b: TOP 15 FEATURE IMPORTANCES (Random Forest)
    # ==========================================
    print("\n=== EXTRACTING RANDOM FOREST FEATURE IMPORTANCES ===")
    # Extract native tree importances
    importances_rf = best_models["Random Forest"].named_steps['classifier'].feature_importances_
    indices_rf = np.argsort(importances_rf)[::-1][:15] # Grab the top 15

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances_rf[indices_rf], y=np.array(ALL_FEATURES)[indices_rf], palette="viridis")
    plt.title("Top 15 Most Diagnostic FRA Features (Random Forest)")
    plt.xlabel("Relative Importance (Gini/Gain)")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig('Plot_8b_Feature_Importance_RF.png', dpi=300)
    print("  -> RF Feature Importances saved as Plot 8b.")


    # ==========================================
    # PLOT 8c: FEATURE IMPORTANCES (SVM via Permutation)
    # ==========================================
    print("\n=== EXTRACTING SVM FEATURE IMPORTANCES (Permutation) ===")
    from sklearn.inspection import permutation_importance

    # RBF SVM lacks native importances, so we calculate the accuracy drop when features are shuffled.
    # Note: n_jobs=-1 uses all CPU cores to speed this up, as it requires re-evaluating the test set.
    svm_model = best_models["SVM (RBF)"]
    
    print("  -> Shuffling features to calculate SVM reliance (This takes a few seconds)...")
    perm_results = permutation_importance(svm_model, X_test_full, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    importances_svm = perm_results.importances_mean
    indices_svm = np.argsort(importances_svm)[::-1][:15]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances_svm[indices_svm], y=np.array(ALL_FEATURES)[indices_svm], palette="flare")
    plt.title("Top 15 Most Diagnostic FRA Features (SVM Permutation Importance)")
    plt.xlabel("Mean Accuracy Decrease (When feature is shuffled)")
    plt.ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig('Plot_8c_Feature_Importance_SVM.png', dpi=300)
    print("  -> SVM Permutation Importances saved as Plot 8c.")

