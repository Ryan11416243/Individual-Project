# Machine Learning for Power Transformer Condition Monitoring (FRA)

This repository contains the complete simulation, feature extraction, and machine learning classification pipeline for automating Frequency Response Analysis (FRA) diagnostics on power transformers. The project utilizes a custom physical simulation engine to generate synthetic fault data, engineers frequency-band features, and trains robust classifiers to identify mechanical and electrical deformations.

## System Architecture

The pipeline is decoupled into three sequential stages to prevent data leakage and ensure reproducibility.

### 1. Physics-Based Data Generation 
The simulation script acts as a high-throughput computational engine, generating realistic transformer FRA traces based on established electromagnetic RLC lumped-parameter models[cite: 4]. 
* **Fault Injection:** Mathematically models four distinct fault geometries: Radial Deformation, Axial Displacement, Loss of Clamping Pressure (LCP), and Turn-to-Turn Short Circuits (TTSC)[cite: 4].
* **Stochastic Realism:** Applies operational variance (thermal drift) and physical measurement noise mimicking Vector Network Analyzer (VNA) instrumentation to ensure realistic datasets[cite: 4].
* **Nodal Admittance Solver:** Constructs and resolves the complex matrix at each frequency step to export raw magnitude and phase data into a structured `Dataset` directory[cite: 4].

### 2. Feature Extraction Pipeline
The extraction script acts as a directory crawler that transforms raw simulation traces into an ML-ready tabular dataset[cite: 3].
* **Statistical Indices:** Calculates critical FRA indicators (CCF, LCC, SDA, SE, CSD) across specifically tuned Low, Medium, and High-Frequency macro-bands[cite: 3].
* **Sub-Band Analysis:** Implements localized windowing to extract resonant peak shifts and integrated spectral energy deviations[cite: 3].
* **Damping Compensation:** Utilizes a polynomial baseline subtraction to isolate physical resonances from background dielectric damping effects[cite: 3].
* **Export:** Compiles all calculated features alongside their physical metadata into `Master_ML_Dataset.csv` while enforcing strict mathematical validation checks[cite: 3].

### 3. Machine Learning Classification
The machine learning pipeline evaluates the diagnostic capability of various supervised learning algorithms on the extracted feature space[cite: 5]. 
* **Leakage-Free Validation:** Employs `StratifiedGroupKFold` cross-validation grouped by the transformer's unique `Unit_ID` to strictly prevent baseline memorization[cite: 5].
* **Model Optimization:** Automates hyperparameter tuning via `RandomizedSearchCV` for Random Forest, Gradient Boosting, and SVM (RBF) classifiers[cite: 5].
* **Diagnostic Visualization:** Automatically generates comprehensive analytical plots, including feature ablation studies, data scalability matrices (fleet width vs. historical depth), normalized confusion matrices, and t-SNE high-dimensional manifolds[cite: 5].

## Dependencies

The codebase requires standard data science and scientific computing libraries. 

* `numpy`
* `pandas`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

## Usage Instructions

To execute the full diagnostic framework, run the scripts sequentially from the root directory:

1. **Generate the Fleet Data:** Run the simulation engine to generate the raw text traces.
   `python multi_gen.py`
2. **Extract Features:** Process the generated `Dataset` directory into the master CSV file.
   `python extraction_sim.py`
3. **Train and Evaluate:** Run the ML experiments and generate the analytical plots.
   `python multimodel.py`
