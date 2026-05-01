# Individual-Project
# Machine Learning for Power Transformer Condition Monitoring (FRA)
This repository contains the complete simulation, feature extraction, and machine learning classification pipeline for automating Frequency Response Analysis (FRA) diagnostics on power transformers. The project utilizes a custom physical simulation engine to generate synthetic fault data, engineers frequency-band features, and trains robust classifiers to identify mechanical and electrical deformations.

# System Architecture
The pipeline is decoupled into three sequential stages to prevent data leakage and ensure reproducibility.

1. Physics-Based Data Generation
The simulation script acts as a high-throughput computational engine, generating realistic transformer FRA traces based on established electromagnetic RLC lumped-parameter models.  

Fault Injection: Mathematically models four distinct fault geometries: Radial Deformation, Axial Displacement, Loss of Clamping Pressure (LCP), and Turn-to-Turn Short Circuits (TTSC).  

Stochastic Realism: Applies operational variance (thermal drift) and physical measurement noise mimicking Vector Network Analyzer (VNA) instrumentation to ensure realistic datasets.  

Nodal Admittance Solver: Constructs and resolves the complex matrix at each frequency step to export raw magnitude and phase data into a structured Dataset directory.  

2. Feature Extraction Pipeline
The extraction script acts as a directory crawler that transforms raw simulation traces into an ML-ready tabular dataset.  

Statistical Indices: Calculates critical FRA indicators (CCF, LCC, SDA, SE, CSD) across specifically tuned Low, Medium, and High-Frequency macro-bands.  

Sub-Band Analysis: Implements localized windowing to extract resonant peak shifts and integrated spectral energy deviations.  

Damping Compensation: Utilizes a polynomial baseline subtraction to isolate physical resonances from background dielectric damping effects.  

Export: Compiles all calculated features alongside their physical metadata into Master_ML_Dataset.csv while enforcing strict mathematical validation checks.  

3. Machine Learning Classification
The machine learning pipeline evaluates the diagnostic capability of various supervised learning algorithms on the extracted feature space.  

Leakage-Free Validation: Employs StratifiedGroupKFold cross-validation grouped by the transformer's unique Unit_ID to strictly prevent baseline memorization.  

Model Optimization: Automates hyperparameter tuning via RandomizedSearchCV for Random Forest, Gradient Boosting, and SVM (RBF) classifiers.  

Diagnostic Visualization: Automatically generates comprehensive analytical plots, including feature ablation studies, data scalability matrices (fleet width vs. historical depth), normalized confusion matrices, and t-SNE high-dimensional manifolds.  

Dependencies
The codebase requires standard data science and scientific computing libraries.

numpy

pandas

scipy

scikit-learn

matplotlib

seaborn

Usage Instructions
To execute the full diagnostic framework, run the scripts sequentially from the root directory:

Generate the Fleet Data: Run the simulation engine to generate the raw text traces.
python multi_gen.py

Extract Features: Process the generated Dataset directory into the master CSV file.
python extraction_sim.py

Train and Evaluate: Run the ML experiments and generate the analytical plots.
python multimodel.py
