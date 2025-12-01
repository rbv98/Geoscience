# Pore Pressure Prediction Using Machine Learning

## Project Purpose

This project develops a hybrid meta-ensemble framework for predicting pore pressure in oil and gas drilling operations. The goal is to prevent costly drilling disasters by accurately predicting formation pore pressure using well log data and machine learning techniques. The project analyzes 284,930 well log samples from 21 wells in Pakistan's Potwar Basin to achieve robust cross-well generalization.

**Course:** CS 7980  
**PI:** Dr. Tehmina Amjad  
**Team:** Rohan Benjamin Varghese, Pranav Patel

## Main Components

### `modeling_visualization.ipynb`
This is the main notebook containing the model development pipeline for pore pressure prediction:

1. **Data Loading & Preprocessing**
   - Loads well log data (GR, DT, RHOB, RES, DPHI, OB, HP, TEMP)
   - Implements well-based train/validation/test splitting (15 training, 5 blind test wells)

3. **Model training**
   - **DFNN:** Deep Feedforward Neural Network (128-64-32-1 architecture)
   - **CNN:** 1D Convolutional Neural Network for spatial patterns
   - **RNN (LSTM):** For sequential geological trends
   - **Random Forest:** 100 trees, depth=6
   - **XGBoost:** 150 trees, learning rate=0.02
   - **Transformer:** 2 heads, 3 blocks with positional encodings

4. **Meta-Ensemble Framework**
   - Combines predictions from all 6 models
   - Uses Ridge regression as meta-learner
   - Implements stacking strategy for optimal combination

5. **Model Interpretability**
   - SHAP analysis for global feature importance
   - LIME for local explanations
   - Feature interaction analysis using H-statistics

6. **Evaluation**
   - Performance metrics: R², RMSE, MAE
   - Cross-well validation
   - Geological formation-specific analysis

### `preprocessing_pipeline.py`
This script executes the end-to-end data preparation pipeline, transforming raw well logs into model-ready datasets:

1. **Data Ingestion & Standardization**
   - Standardizes nomenclature (lowercase, stripped spaces) and handles null flags (-999.25)
   - Validates essential log availability (TVD, DT, GR, SPHI, HP, OB)
   - Filters specific depth ranges for Murree formation wells (Missa Keswal, Qazian)

2. **Feature Engineering**
   - **Physics-based features:** Eaton ratio, hydrostatic (`hp_gradient`) and overburden (`ob_gradient`) gradients
   - **Target transformations:** Calculates `pressure_ratio` and `overpressure`
   - **Regime Classification:** Bins pressure ratios into 5 discrete classes (Underpressured to Severe OP)

3. **Quality Control & Selection**
   - Enforces physical bounds to remove outliers (e.g., TVD < 6000, PPP < 30000)
   - Removes target leakage features (MW, DST, LLD)
   - Implements coverage-based feature selection (>65% availability threshold)

4. **Data Splitting & Export**
   - Performs well-based splitting (Train, Validation, and Test sets)
   - Generates metadata JSON tracking feature coverage and split statistics
   - Exports processed datasets as `train_data.csv`, `val_data.csv`, and `test_data.csv`

### `exp/validation_suite.ipynb`
This notebook serves as the experimental foundation and quality assurance layer for the preprocessing pipeline:

1. **Exploratory Data Analysis (EDA)**
   - **Feature Coverage Analysis:** Visualizes data availability across wells to establish the 65% inclusion threshold.
   - **Leakage Detection:** correlation analysis to identify and exclude features with direct target leakage (e.g., MW, DST).
   - **Outlier Definition:** Uses distribution plots to determine physical cutoff values for PPP, HP, and TVD.

2. **Pipeline Logic Verification**
   - **Split Integrity:** Verifies that Test/Validation wells are completely isolated from the Training set.
   - **Engineering Checks:** Validates the calculation logic for Eaton ratio and pressure gradients against theoretical bounds.
   - **Formation filtering:** Confirms the correct depth extraction for Murree formation wells.

3. **Post-Processing Audit**
   - **Distribution Consistency:** Compares raw vs. processed data to ensure cleaning didn't alter geological trends.
   - **Target Analysis:** Verifies the class balance of the engineered "Pressure Regime" variable.

### `exp/m1_tree.ipynb`
This notebook focuses on the development and evaluation of tree-based ensemble methods, specifically XGBoost, to establish a robust baseline:

1. **Baseline Modeling**
   - Implements an initial XGBoost model using the clean features from the preprocessing pipeline.
   - Tests log-transformation of the target variable (`log1p(PPP)`) to handle value range disparities.

2. **Geological Feature Engineering**
   - **Signal Extraction:** Creates features to capture geological anomalies rather than absolute values.
   - **Spatial Context:** Engineers features to better represent the geological trends found in the well logs.

3. **Performance Analysis**
   - **Enhanced Model:** Trains an XGBoost model on the enhanced feature set.
   - **Failure Mode Analysis:** Specifically investigates "Underpressure" prediction failures, diagnosing where the model struggles to distinguish pressure regimes.

### `exp/m2_NN.ipynb`
This notebook explores Deep Learning approaches to capture complex non-linear relationships in the well data:

1. **Data Preparation for DL**
   - Standardizes inputs specifically for neural network convergence.
   - Segregates features and targets suitable for tensor operations.

2. **Deep Feedforward Neural Network (DFNN)**
   - **Architecture:** Implements a multi-layer perceptron with a tapering architecture (256 → 128 → 64 → 32).
   - **Regularization:** Applies Dropout (0.3) and L2 regularization to prevent overfitting on the relatively small well dataset.

3. **Iterative Improvement**
   - **Feature Augmentation:** Introduces specific engineered features designed to aid the DFNN without introducing temporal leakage.
   - **Hyperparameter Tuning:** Adjusts layer depth and regularization strength to optimize validation loss.

### `exp/model_training_kfold.ipynb`
This notebook implements a robust 5-Fold Cross-Validation strategy to maximize training data utilization and ensure fair model evaluation:

1. **Data Consolidation**
   - Merges previous Training and Validation sets into a single `train_val_combined` dataset to increase sample size for the K-Fold process.
   - Retains the blind test set separately for final independent verification.

2. **K-Fold Stacking Architecture**
   - **Base Models:** Generates Out-of-Fold (OOF) predictions for 5 diverse architectures: CNN, DFNN, RNN (LSTM), Random Forest, and XGBoost.
   - **Leakage Prevention:** Uses 5-Fold splitting to ensure base model predictions used for meta-training are strictly generated on unseen data.

3. **Meta-Model Training**
   - **Level-2 Learner:** Trains a meta-model (Ridge Regression/Linear) on the stacked OOF predictions from the 5 base models.
   - **Evaluation:** Assesses performance metrics (R², RMSE) across folds to provide a statistically significant measure of model stability.

## Dependencies

```python
# Core libraries
pandas==1.3.0
numpy==1.21.0
scikit-learn==1.0.2
matplotlib==3.4.0
seaborn==0.11.0

# Deep Learning
tensorflow==2.10.0
keras==2.10.0

# Tree-based models
xgboost==1.6.0

# Interpretability
shap==0.41.0
lime==0.2.0

# Data processing
scipy==1.7.0
```

## How to Run the Code

### 1\. Environment Setup

Ensure you have a Python environment ready with Jupyter installed.

```bash
# Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow xgboost shap lime scipy jupyterlab
```

### 2\. Data Preparation

The pipeline expects raw CSV files to be placed in a specific directory.

1.  Ensure all the well log `.csv` files are inside `./datasets/`.
2.  **Run the preprocessing pipeline:**
    ```bash
    python preprocessing_pipeline.py
    ```
      * **Output:** This will generate three files in your root directory: `train_data.csv`, `val_data.csv`, and `test_data.csv`, along with a `preprocessing_metadata.json` file.

### 3\. Model Training & Analysis

Once the data is processed, you can run the experiments or the main visualization notebook.

1.  **Experiments:** Navigate to the `exp/` folder to run specific model tests (e.g., `m1_tree.ipynb` for XGBoost baselines or `model_training_kfold.ipynb` for the full K-Fold stacking).
2.  **Main Pipeline:** Open `modeling_visualization.ipynb` to see the consolidated results and interpretability analysis.

### Expected Inputs (Raw Data)

The `preprocessing_pipeline.py` script automatically cleans and standardizes the data. Ensure your input CSVs contain the following columns (names are case-insensitive as the script handles standardization):

  * **Essential Features (Must exist in all wells):**

      * `TVD`: True Vertical Depth
      * `DT`: Compressional Sonic Log
      * `DT_NCT`: Normal Compaction Trend for DT
      * `GR`: Gamma Ray
      * `SPHI`: Sonic Porosity
      * `HP`: Hydrostatic Pressure
      * `OB`: Overburden Pressure
      * `PPP`: Pore Pressure (Target Variable)

  * **Optional Features (Used if coverage \> 65%):**

      * `RHOB_COMBINED`: Bulk Density
      * `RES_DEEP`: Deep Resistivity

### Expected Outputs
- **Trained Models:** Saved as `.pkl` or `.h5` files
- **Predictions:** Pore pressure values in PSI
- **Performance Metrics:**
  - R² score: ~0.915
  - RMSE: ~547 psi
  - MAE: ~397 psi
- **Visualizations:**
  - SHAP summary plots
  - Feature importance rankings
  - Prediction vs actual scatter plots
  - Cross-well validation results

## Implementation Attribution

### Rohan Benjamin Varghese
- Model architecture design (DFNN, CNN, RNN, Transformer)
- Meta-ensemble framework implementation
- Model training and hyperparameter optimization
- Interpretability analysis (SHAP and LIME)
- Performance evaluation and metrics computation
- Results visualization

### Pranav Patel
- Designed and implemented the automated preprocessing script for data ingestion, cleaning, and standardization.
- Developed physics-based features (Eaton ratio, pressure gradients) and geological signal extraction algorithms.
- Conducted exploratory analysis to verify depth-slicing logic, feature coverage, and split distribution balance.
- Established robust XGBoost baselines and performed failure mode analysis on underpressure predictions.
- Optimized Deep Feedforward Neural Network (DFNN) architectures and implemented regularization strategies for small datasets.
- Implemented 5-Fold Cross-Validation and the K-Fold stacking architecture for the meta-ensemble framework.


## Results Summary

| Model | R² Score | RMSE (psi) |
|-------|----------|------------|
| DFNN | 0.858 | 705 |
| CNN | 0.878 | 653 |
| RNN | 0.836 | 756 |
| Random Forest | 0.880 | 648 |
| XGBoost | 0.889 | 622 |
| Transformer | 0.878 | 652 |
| **Meta-Ensemble** | **0.915** | **547** |


