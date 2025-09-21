# Usage Guide for Quantitative Competition Pipeline

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify data is in place:**
```bash
ls data/train.csv  # Should show the training data file
```

3. **Check installation:**
```python
python -c "import lightgbm, xgboost, torch; print('All packages installed!')"
```

## 📊 Running the Pipeline

### Option 1: Full Automated Pipeline (Recommended)
```bash
python src/train_ensemble.py
```
This runs the complete 48-hour pipeline with all models and ensemble optimization.

### Option 2: Baseline Model Only
```bash
python src/train.py
```
This trains only the LightGBM baseline model with cross-validation.

### Option 3: Interactive EDA
```bash
jupyter notebook notebooks/eda.ipynb
```
Explore the data interactively before training.

## 🔧 Custom Training Workflows

### Training Individual Models

```python
from src.utils.cross_validation import PurgedTimeSeriesCV
from src.features.feature_engineer import FeatureEngineer
from src.models.lgbm_model import LightGBMModel
import pandas as pd

# Load data
df = pd.read_csv('data/train.csv')
X = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']]
y = df[['Y1', 'Y2']]

# Setup cross-validation
cv = PurgedTimeSeriesCV(n_splits=3, gap_size=4000)

# Feature engineering
fe = FeatureEngineer(n_top_interactions=35)
X_fe = fe.fit_transform(X, y)

# Train model
model = LightGBMModel()
cv_scores, oof_predictions = model.cross_validate(X_fe, y, cv)
```

### Creating Custom Ensembles

```python
from src.ensemble import MiddleAveragingEnsemble, WeightedEnsemble

# Assuming you have predictions from different models
predictions = {
    'lgbm': lgbm_predictions,
    'xgboost': xgb_predictions,
    'neural_net': nn_predictions
}

# Middle averaging (recommended)
middle_ens = MiddleAveragingEnsemble(keep_ratio=0.6)
ensemble_pred = middle_ens.predict(predictions)

# Or weighted ensemble
weighted_ens = WeightedEnsemble()
weights = weighted_ens.optimize_weights(predictions, y_true)
ensemble_pred = weighted_ens.predict(predictions)
```

## ⚙️ Configuration

### Modifying Hyperparameters

Edit `src/config.py` to adjust model parameters:

```python
LGBM_PARAMS = {
    'n_estimators': 300,  # Increase for more trees
    'max_depth': 6,       # Increase for deeper trees
    'learning_rate': 0.03,  # Decrease for better generalization
    # ... other parameters
}
```

### Adjusting Feature Engineering

```python
FE_PARAMS = {
    'n_top_interactions': 35,  # Number of interaction features (30-40 recommended)
    'rolling_window': 150,      # Window for rolling statistics
    'min_correlation': 0.1      # Minimum correlation threshold
}
```

### Cross-Validation Settings

```python
CV_PARAMS = {
    'n_splits': 3,
    'gap_size': 4000,  # CRITICAL: Don't reduce below 4000 (5% of data)
    'val_size': 10000
}
```

## 📈 Performance Monitoring

### Understanding CV Output

The pipeline provides detailed metrics:
```
Fold 1 Results:
  Y1 R²: 0.7234
  Y2 R²: 0.6891
  Weighted R²: 0.7063  ← This is your key metric

Cross-Validation Summary:
  Y1: 0.7156 ± 0.0145
  Y2: 0.6823 ± 0.0201
  Weighted: 0.6989 ± 0.0173  ← Must be > 0.68 to proceed
```

### Go/No-Go Gates

The pipeline has automatic decision points:

1. **Baseline Gate (CV > 0.68)**: Pipeline stops if not met
2. **Diversity Check**: XGBoost correlation with LightGBM should be < 0.9
3. **Ensemble Improvement**: Should add >0.02 to best single model
4. **Holdout Validation**: Gap between CV and holdout should be < 0.03

## 🐛 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| "CV score below 0.68" | 1. Check data quality<br>2. Increase feature interactions<br>3. Tune hyperparameters<br>4. Verify no data leakage |
| "High train-val gap" | 1. Increase regularization<br>2. Reduce model complexity<br>3. Add more dropout (NN)<br>4. Check for overfitting |
| "Models too correlated" | 1. Use different hyperparameters<br>2. Try different feature subsets<br>3. Change random seeds |
| "Out of memory" | 1. Reduce batch size (NN)<br>2. Reduce n_estimators<br>3. Use subset of features |

### Validation Checks

Run these checks to ensure pipeline integrity:

```python
# Check for temporal leakage
from src.utils.cross_validation import PurgedTimeSeriesCV
cv = PurgedTimeSeriesCV()
cv.validate_temporal_integrity(X, y)  # Should pass all assertions

# Verify feature engineering
from src.features.feature_engineer import validate_no_leakage
validate_no_leakage(X_train, X_val, train_idx, val_idx)  # Should return True
```

## 🎯 Competition Timeline

### Hour-by-Hour Strategy

| Hours | Task | Command/Action |
|-------|------|----------------|
| 0-2 | Setup & EDA | `jupyter notebook notebooks/eda.ipynb` |
| 2-8 | Baseline training | `python src/train.py` |
| 8-10 | Verify CV > 0.68 | Check output, tune if needed |
| 10-14 | XGBoost training | Continue pipeline |
| 14-18 | Neural network | Automatic if time permits |
| 18-24 | Ensemble optimization | Automatic in pipeline |
| 24-30 | Validation & tuning | Monitor holdout performance |
| 30-36 | Final model training | Train on full data |
| 36-42 | Submission prep | Generate predictions |
| 42-48 | Buffer & documentation | Handle any issues |

## 💾 Output Files

After running the pipeline, you'll find:

```
models/
├── final_lgbm_y1.txt      # LightGBM model for Y1
├── final_lgbm_y2.txt      # LightGBM model for Y2
├── final_xgb_y1.json      # XGBoost model for Y1
├── final_xgb_y2.json      # XGBoost model for Y2
└── ensemble_models.pkl    # Ensemble configurations

results/
├── baseline_lgbm_results.json    # CV scores and parameters
├── ensemble_results.json          # Final ensemble performance
├── oof_lgbm.npy                  # Out-of-fold predictions
├── oof_xgboost.npy               # Out-of-fold predictions
└── oof_neural_net.npy            # Out-of-fold predictions
```

## 🔍 Advanced Features

### Custom Cross-Validation

```python
# Create custom splits
cv = PurgedTimeSeriesCV(n_splits=5, gap_size=2000)
for train_idx, val_idx in cv.split(X):
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
```

### Feature Importance Analysis

```python
# After training
importance_df = model.get_feature_importance_summary(top_n=20)
print(importance_df)
```

### Ensemble Comparison

```python
from src.ensemble import compare_ensemble_methods
comparison_df = compare_ensemble_methods(predictions, y_true)
# Shows performance of different ensemble strategies
```

## 📞 Getting Help

1. **Check logs**: All operations are logged with timestamps
2. **Review configurations**: `src/config.py` has all settings
3. **Validate data**: Use the EDA notebook to check data quality
4. **Test components individually**: Each module can be run standalone

## ⚠️ Critical Warnings

1. **NEVER use standard K-fold CV** - Will cause temporal leakage
2. **NEVER reduce gap below 4000 samples** - 5% gap is minimum
3. **NEVER skip feature engineering validation** - Leakage is subtle
4. **ALWAYS check train-val gap** - >0.05 indicates overfitting
5. **ALWAYS verify model diversity** - Correlation >0.9 reduces ensemble benefit

## 🎉 Success Criteria

Your pipeline is successful when:
- ✅ Baseline CV score > 0.68
- ✅ Train-validation gap < 0.05
- ✅ Model correlation < 0.9
- ✅ Ensemble improvement > 0.02
- ✅ Holdout within 0.03 of CV score