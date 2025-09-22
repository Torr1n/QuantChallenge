"""
Advanced training script with temporal group separation.

Implements separate handling for time-series and non-time-series features
based on ACF analysis showing 96-unit seasonality.
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
from src.features.feature_engineer_v2 import AdvancedFeatureEngineer
from src.models.lgbm_model import LightGBMModel


def load_data():
    """Load and prepare the training data."""
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    # Load data
    df = pd.read_csv(TRAIN_FILE)
    print(f"Data shape: {df.shape}")

    # Separate features and targets
    X = df[FEATURE_COLS]
    y = df[TARGET_COLS]

    # Basic data info
    print(f"\nFeatures shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # Check target correlation
    target_corr = y.corr().iloc[0, 1]
    print(f"\nY1-Y2 correlation: {target_corr:.4f}")

    # Print feature groups
    print("\n📊 Feature Groups (based on ACF analysis):")
    print(f"  Non-temporal (→Y1): {AdvancedFeatureEngineer.NON_TEMPORAL_FEATURES}")
    print(f"  Temporal 96-cycle (→Y2): {AdvancedFeatureEngineer.TEMPORAL_FEATURES}")

    return X, y


def train_baseline_model_v2():
    """Train LightGBM baseline with advanced feature engineering."""
    print("\n" + "="*60)
    print("Training Advanced LightGBM Model")
    print("="*60)

    # Load data
    X, y = load_data()

    # Initialize cross-validation
    cv = PurgedTimeSeriesCV(**CV_PARAMS)

    # Validate CV integrity
    print("\n" + "="*60)
    print("Validating Cross-Validation Setup")
    print("="*60)
    cv.validate_temporal_integrity(X, y)

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

        # Feature engineering with temporal awareness
        print("\nEngineering features with temporal group separation...")
        start_time = time.time()

        # Use advanced feature engineer with fold index for buffer management
        fe = AdvancedFeatureEngineer(
            n_top_interactions=35,
            rolling_window=96,  # Critical: matches seasonality period
            min_correlation=0.1
        )

        X_train_fe = fe.fit_transform(X_train, y_train, fold_idx=fold_idx)
        X_val_fe = fe.transform(X_val, fold_idx=fold_idx)

        fe_time = time.time() - start_time
        print(f"Feature engineering completed in {fe_time:.1f}s")

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
        print(f"  Y1 R² (non-temporal target): {r2_y1:.4f}")
        print(f"  Y2 R² (temporal target): {r2_y2:.4f}")
        print(f"  Weighted R²: {weighted_r2:.4f}")
        print(f"  Train-val gaps: Y1={cv_scores[-1]['train_val_gap_y1']:.4f}, Y2={cv_scores[-1]['train_val_gap_y2']:.4f}")

        # Get feature importance for first fold
        if fold_idx == 0:
            print("\nFeature Importance (Top 15):")
            importance_df = model.get_feature_importance_summary(top_n=15)

            # Analyze temporal vs non-temporal feature importance
            temporal_importance = 0
            non_temporal_importance = 0

            for _, row in importance_df.iterrows():
                feat_name = row['feature']
                # Check if feature is from temporal group
                is_temporal = any(tf in feat_name for tf in AdvancedFeatureEngineer.TEMPORAL_FEATURES)
                if is_temporal:
                    temporal_importance += row['avg_importance']
                else:
                    non_temporal_importance += row['avg_importance']

            print(f"\n📊 Feature Group Importance:")
            print(f"  Temporal features: {temporal_importance:.1f} ({temporal_importance/(temporal_importance+non_temporal_importance)*100:.1f}%)")
            print(f"  Non-temporal features: {non_temporal_importance:.1f} ({non_temporal_importance/(temporal_importance+non_temporal_importance)*100:.1f}%)")

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
    print(f"Y1 R² (non-temporal): {avg_scores['r2_y1_mean']:.4f} ± {avg_scores['r2_y1_std']:.4f}")
    print(f"Y2 R² (temporal): {avg_scores['r2_y2_mean']:.4f} ± {avg_scores['r2_y2_std']:.4f}")
    print(f"Weighted R²: {avg_scores['weighted_r2_mean']:.4f} ± {avg_scores['weighted_r2_std']:.4f}")
    print(f"Average train-val gap: {avg_scores['avg_train_val_gap']:.4f}")

    # Performance analysis by target type
    print(f"\n📊 Performance by Target Type:")
    if avg_scores['r2_y1_mean'] > avg_scores['r2_y2_mean']:
        print(f"  ✓ Y1 (non-temporal) performing better: +{(avg_scores['r2_y1_mean'] - avg_scores['r2_y2_mean']):.4f}")
        print(f"    Non-temporal features are well captured")
    else:
        print(f"  ✓ Y2 (temporal) performing better: +{(avg_scores['r2_y2_mean'] - avg_scores['r2_y1_mean']):.4f}")
        print(f"    96-unit seasonality features are effective")

    # Check if we meet the threshold
    if avg_scores['weighted_r2_mean'] >= MIN_CV_SCORE:
        print(f"\n✅ SUCCESS: Advanced CV score {avg_scores['weighted_r2_mean']:.4f} exceeds {MIN_CV_SCORE} threshold!")
        print("Ready to proceed with ensemble models.")
    else:
        print(f"\n⚠️ WARNING: Advanced CV score {avg_scores['weighted_r2_mean']:.4f} below {MIN_CV_SCORE} threshold!")

    # Save results
    results = {
        'cv_scores': cv_scores,
        'avg_scores': avg_scores,
        'model_params': LGBM_PARAMS,
        'cv_params': CV_PARAMS,
        'feature_engineering': {
            'rolling_window': 96,
            'seasonality_period': 96,
            'n_interactions': 35,
            'temporal_features': AdvancedFeatureEngineer.TEMPORAL_FEATURES,
            'non_temporal_features': AdvancedFeatureEngineer.NON_TEMPORAL_FEATURES
        }
    }

    results_path = RESULTS_DIR / 'advanced_lgbm_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save OOF predictions
    oof_path = RESULTS_DIR / 'advanced_lgbm_oof.npy'
    np.save(oof_path, oof_predictions)
    print(f"OOF predictions saved to {oof_path}")

    return avg_scores, oof_predictions


def train_final_model_v2():
    """Train final model with advanced features on all data."""
    print("\n" + "="*60)
    print("Training Final Advanced Model on All Data")
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
    print("\nEngineering features with temporal awareness...")
    fe = AdvancedFeatureEngineer(
        n_top_interactions=35,
        rolling_window=96,
        min_correlation=0.1
    )
    X_train_fe = fe.fit_transform(X_train, y_train, fold_idx=0)
    X_holdout_fe = fe.transform(X_holdout, fold_idx=0)

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
    print(f"  Y1 R² (non-temporal): {r2_y1:.4f}")
    print(f"  Y2 R² (temporal): {r2_y2:.4f}")
    print(f"  Weighted R²: {weighted_r2:.4f}")

    # Save final model
    model_path = MODELS_DIR / 'final_advanced_lgbm'
    model.save_models(str(model_path))
    print(f"\nFinal model saved to {model_path}")

    return model, fe


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADVANCED QUANTITATIVE COMPETITION TRAINING PIPELINE")
    print("With Temporal Group Separation (96-unit seasonality)")
    print("="*60)

    # Train advanced baseline model with CV
    avg_scores, oof_predictions = train_baseline_model_v2()

    # If baseline successful, train final model
    if avg_scores['weighted_r2_mean'] >= MIN_CV_SCORE:
        print("\nProceeding with final model training...")
        train_final_model_v2()
    else:
        print("\nBaseline performance insufficient. Please improve before proceeding.")