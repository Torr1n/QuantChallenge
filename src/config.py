"""
Global configuration for the quantitative competition pipeline.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for dir_path in [MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Data files
TRAIN_FILE = DATA_DIR / "train.csv"

# Column names
FEATURE_COLS = [chr(ord('A') + i) for i in range(14)]  # A through N (14 features)
TARGET_COLS = ['Y1', 'Y2']
TIME_COL = 'time'
ALL_FEATURE_COLS = FEATURE_COLS  # Will be extended with engineered features

# Model parameters - Proven from Jane Street analysis
LGBM_PARAMS = {
    'objective': 'regression',
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,     # L1 regularization
    'reg_lambda': 1.0,    # L2 regularization
    'min_child_samples': 50,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 250,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.05,
    'reg_lambda': 1.5,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Neural Network parameters
NN_PARAMS = {
    'layers': [32, 64, 32],
    'dropout': 0.4,  # High dropout critical for preventing overfitting
    'activation': 'relu',
    'batch_size': 512,
    'epochs': 100,
    'early_stopping_patience': 10,
    'learning_rate': 0.001,
    'weight_decay': 1e-4
}

# Cross-validation parameters
CV_PARAMS = {
    'n_splits': 3,
    'gap_size': 4000,  # 5% gap between train and validation
    'val_size': 10000
}

# Feature engineering parameters
FE_PARAMS = {
    'n_top_interactions': 35,  # Top interaction features to select
    'rolling_window': 150,      # Window for rolling statistics
    'min_correlation': 0.1      # Minimum correlation with target for feature selection
}

# Ensemble parameters
ENSEMBLE_PARAMS = {
    'middle_ratio': 0.6,  # Use middle 60% of predictions
    'min_weight': 0.1,    # Minimum weight for any model
    'max_weight': 0.5     # Maximum weight for any model
}

# Training settings
RANDOM_SEED = 42
USE_GPU = False  # Set to True if GPU available for neural networks

# Validation thresholds
MIN_CV_SCORE = 0.68  # Minimum CV score for baseline before proceeding
MAX_TRAIN_VAL_GAP = 0.05  # Maximum acceptable gap between train and validation scores

# Logging
VERBOSE = True