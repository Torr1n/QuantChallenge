"""
Main training script for the quantitative competition pipeline.

Orchestrates data loading, feature engineering, model training, and validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import warnings
warnings.filterwarnings('ignore')

from src.config import *
from src.utils.cross_validation import PurgedTimeSeriesCV
from src.features.feature_engineer import FeatureEngineer, validate_no_leakage
from src.models.lgbm_model import LightGBMModel


def load_data():
    """Load and prepare the training data."""
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    # Load data
    df = pd.read_csv(TRAIN_FILE)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Separate features and targets
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    # Basic data info
    print(f"\nFeatures shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Missing values in features: {X.isna().sum().sum()}")
    print(f"Missing values in targets: {y.isna().sum().sum()}")

    # Check target correlation
    target_corr = y.corr().iloc[0, 1]
    print(f"\nY1-Y2 correlation: {target_corr:.4f}")

    if abs(target_corr) < 0.3:
        print("⚠️ Warning: Low correlation between targets. Consider independent models.")
    elif abs(target_corr) > 0.9:
        print("⚠️ Warning: Very high correlation between targets. Check for redundancy.")

    return X, y


def perform_eda(X, y):
    """Perform basic exploratory data analysis."""
    print("\n" + "="*60)
    print("Exploratory Data Analysis")
    print("="*60)

    # Feature statistics
    print("\nFeature Statistics:")
    print(X.describe().T[['mean', 'std', 'min', 'max']])

    # Target statistics
    print("\nTarget Statistics:")
    print(y.describe().T)

    # Check for spike features (high concentration of specific values)
    print("\nChecking for spike features...")
    spike_features = []
    for col in X.columns:
        value_counts = X[col].value_counts()
        if len(value_counts) > 0:
            top_value_ratio = value_counts.iloc[0] / len(X)
            if top_value_ratio > 0.1:  # More than 10% same value
                spike_features.append(col)
                print(f"  {col}: {top_value_ratio:.1%} samples have value {value_counts.index[0]:.4f}")

    if spike_features:
        print(f"Found {len(spike_features)} spike features: {spike_features}")

    return spike_features


def train_baseline_model():
    """Train the LightGBM baseline model with cross-validation."""
    print("\n" + "="*60)
    print("Training LightGBM Baseline Model")
    print("="*60)

    # Load data
    X, y = load_data()

    # Perform EDA
    spike_features = perform_eda(X, y)

    # Initialize cross-validation
    cv = PurgedTimeSeriesCV(**CV_PARAMS)

    # Validate CV integrity
    print("\n" + "="*60)
    print("Validating Cross-Validation Setup")
    print("="*60)
    cv.validate_temporal_integrity(X, y)

    # Initialize feature engineer
    fe = FeatureEngineer(**FE_PARAMS)

    # Store out-of-fold predictions
    oof_predictions = np.zeros((len(X), 2))
    cv_scores = []

    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{cv.get_n_splits()}")
        print(f"{'='*60}")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train: {len(X_train)} samples [{train_idx[0]}:{train_idx[-1]}]")
        print(f"Val: {len(X_val)} samples [{val_idx[0]}:{val_idx[-1]}]")

        # Feature engineering
        print("\nEngineering features...")
        start_time = time.time()

        if fold_idx == 0:
            # Fit on first fold
            X_train_fe = fe.fit_transform(X_train, y_train)
        else:
            # Re-fit for each fold (important for preventing leakage)
            fe = FeatureEngineer(**FE_PARAMS)
            X_train_fe = fe.fit_transform(X_train, y_train)

        X_val_fe = fe.transform(X_val)

        fe_time = time.time() - start_time
        print(f"Feature engineering completed in {fe_time:.1f}s")

        # Validate no leakage
        validate_no_leakage(X_train_fe, X_val_fe, train_idx, val_idx)

        # Train model
        print("\nTraining LightGBM model...")
        start_time = time.time()

        model = LightGBMModel(params=LGBM_PARAMS)
        model.fit(X_train_fe, y_train, X_val_fe, y_val, early_stopping_rounds=10)

        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.1f}s")

        # Generate predictions
        val_pred = model.predict(X_val_fe)
        oof_predictions[val_idx] = val_pred

        # Calculate scores
        from sklearn.metrics import r2_score
        r2_y1 = r2_score(y_val['Y1'], val_pred[:, 0])
        r2_y2 = r2_score(y_val['Y2'], val_pred[:, 1])
        weighted_r2 = model.get_weighted_r2_score(y_val, val_pred)

        cv_scores.append({
            'fold': fold_idx + 1,
            'r2_y1': r2_y1,
            'r2_y2': r2_y2,
            'weighted_r2': weighted_r2,
            'train_val_gap_y1': model.train_scores['Y1']['r2'] - model.val_scores['Y1']['r2'],
            'train_val_gap_y2': model.train_scores['Y2']['r2'] - model.val_scores['Y2']['r2']
        })

        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Y1 R²: {r2_y1:.4f} (train-val gap: {cv_scores[-1]['train_val_gap_y1']:.4f})")
        print(f"  Y2 R²: {r2_y2:.4f} (train-val gap: {cv_scores[-1]['train_val_gap_y2']:.4f})")
        print(f"  Weighted R²: {weighted_r2:.4f}")

        # Check for overfitting
        if cv_scores[-1]['train_val_gap_y1'] > MAX_TRAIN_VAL_GAP:
            print(f"⚠️ Warning: Y1 overfitting detected (gap: {cv_scores[-1]['train_val_gap_y1']:.4f})")

        if cv_scores[-1]['train_val_gap_y2'] > MAX_TRAIN_VAL_GAP:
            print(f"⚠️ Warning: Y2 overfitting detected (gap: {cv_scores[-1]['train_val_gap_y2']:.4f})")

        # Get feature importance for first fold
        if fold_idx == 0:
            print("\nFeature Importance (Top 10):")
            importance_df = model.get_feature_importance_summary(top_n=10)

    # Calculate average CV scores
    cv_scores_df = pd.DataFrame(cv_scores)
    avg_scores = {
        'r2_y1_mean': cv_scores_df['r2_y1'].mean(),
        'r2_y1_std': cv_scores_df['r2_y1'].std(),
        'r2_y2_mean': cv_scores_df['r2_y2'].mean(),
        'r2_y2_std': cv_scores_df['r2_y2'].std(),
        'weighted_r2_mean': cv_scores_df['weighted_r2'].mean(),
        'weighted_r2_std': cv_scores_df['weighted_r2'].std(),
        'avg_train_val_gap': cv_scores_df[['train_val_gap_y1', 'train_val_gap_y2']].mean().mean()
    }

    # Print final summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"Y1 R²: {avg_scores['r2_y1_mean']:.4f} ± {avg_scores['r2_y1_std']:.4f}")
    print(f"Y2 R²: {avg_scores['r2_y2_mean']:.4f} ± {avg_scores['r2_y2_std']:.4f}")
    print(f"Weighted R²: {avg_scores['weighted_r2_mean']:.4f} ± {avg_scores['weighted_r2_std']:.4f}")
    print(f"Average train-val gap: {avg_scores['avg_train_val_gap']:.4f}")

    # Check if we meet the threshold
    if avg_scores['weighted_r2_mean'] >= MIN_CV_SCORE:
        print(f"\n✅ SUCCESS: Baseline CV score {avg_scores['weighted_r2_mean']:.4f} exceeds {MIN_CV_SCORE} threshold!")
        print("Ready to proceed with ensemble models.")
    else:
        print(f"\n⚠️ WARNING: Baseline CV score {avg_scores['weighted_r2_mean']:.4f} below {MIN_CV_SCORE} threshold!")
        print("Consider:")
        print("  - More feature engineering")
        print("  - Hyperparameter tuning")
        print("  - Checking for data quality issues")

    # Save results
    results = {
        'cv_scores': cv_scores,
        'avg_scores': avg_scores,
        'model_params': LGBM_PARAMS,
        'cv_params': CV_PARAMS,
        'fe_params': FE_PARAMS
    }

    results_path = RESULTS_DIR / 'baseline_lgbm_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save OOF predictions
    oof_path = RESULTS_DIR / 'baseline_lgbm_oof.npy'
    np.save(oof_path, oof_predictions)
    print(f"OOF predictions saved to {oof_path}")

    return avg_scores, oof_predictions


def train_final_model():
    """Train final model on all data for submission."""
    print("\n" + "="*60)
    print("Training Final Model on All Data")
    print("="*60)

    # Load all data
    X, y = load_data()

    # Get holdout indices for final validation
    cv = PurgedTimeSeriesCV(**CV_PARAMS)
    holdout_idx = cv.get_holdout_indices(len(X))

    # Split into train and holdout
    train_idx = np.arange(0, holdout_idx[0])
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]

    print(f"Training on {len(X_train)} samples")
    print(f"Holdout validation on {len(X_holdout)} samples")

    # Feature engineering
    print("\nEngineering features...")
    fe = FeatureEngineer(**FE_PARAMS)
    X_train_fe = fe.fit_transform(X_train, y_train)
    X_holdout_fe = fe.transform(X_holdout)

    # Train final model
    print("\nTraining final model...")
    model = LightGBMModel(params=LGBM_PARAMS)
    model.fit(X_train_fe, y_train)

    # Validate on holdout
    holdout_pred = model.predict(X_holdout_fe)

    from sklearn.metrics import r2_score
    r2_y1 = r2_score(y_holdout['Y1'], holdout_pred[:, 0])
    r2_y2 = r2_score(y_holdout['Y2'], holdout_pred[:, 1])
    weighted_r2 = model.get_weighted_r2_score(y_holdout, holdout_pred)

    print(f"\nHoldout Results:")
    print(f"  Y1 R²: {r2_y1:.4f}")
    print(f"  Y2 R²: {r2_y2:.4f}")
    print(f"  Weighted R²: {weighted_r2:.4f}")

    # Save final model
    model_path = MODELS_DIR / 'final_lgbm'
    model.save_models(str(model_path))
    print(f"\nFinal model saved to {model_path}")

    return model, fe


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTITATIVE COMPETITION TRAINING PIPELINE")
    print("="*60)

    # Train baseline model with CV
    avg_scores, oof_predictions = train_baseline_model()

    # If baseline successful, train final model
    if avg_scores['weighted_r2_mean'] >= MIN_CV_SCORE:
        print("\nProceeding with final model training...")
        train_final_model()
    else:
        print("\nBaseline performance insufficient. Please improve before proceeding.")