# Pore Pressure Prediction Using Machine Learning

## Project Purpose

This project develops a hybrid meta-ensemble framework for predicting pore pressure in oil and gas drilling operations. The goal is to prevent costly drilling disasters by accurately predicting formation pore pressure using well log data and machine learning techniques. The project analyzes 284,930 well log samples from 21 wells in Pakistan's Potwar Basin to achieve robust cross-well generalization.

**Course:** CS 7980  
**PI:** Dr. Tehmina Amjad  
**Team:** Rohan Benjamin Varghese, Pranav Patel

## Main Components

### `model_cleaned.ipynb`
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

### Setup
1. Install Jupyter Notebook or JupyterLab
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow xgboost shap lime scipy
   ```

### Execution Steps
1. Open `model_cleaned.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook is divided into clear sections - you can run each section independently after initial data loading

### Expected Inputs
- **Input Format:** CSV file with well log data
- **Required Columns:** 
  - Well identifiers (Well_ID)
  - Depth measurements (TVD)
  - Core logs: GR, DT, RHOB, RES, DPHI
  - Pressure measurements: OB (overburden), HP (hydrostatic), PPP (target)
  - Temperature (TEMP)

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


