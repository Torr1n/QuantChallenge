"""
XGBoost Multi-Target Model Implementation

Diversification model for ensemble with different hyperparameters and tree construction.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class XGBoostModel:
    """
    XGBoost model with multi-target support for ensemble diversity.

    Uses different tree construction method and regularization approach than LightGBM.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.

        Args:
            params: Model parameters dictionary (uses defaults if None)
        """
        self.default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 250,
            'max_depth': 5,  # Shallower than LightGBM for diversity
            'learning_rate': 0.05,
            'subsample': 0.8,  # More aggressive sampling
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,  # L1 regularization
            'reg_lambda': 1.5,  # L2 regularization
            'min_child_weight': 5,
            'gamma': 0.1,  # Minimum loss reduction (unique to XGBoost)
            'tree_method': 'hist',  # Different from LightGBM's method
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
            'early_stopping_rounds': 10,  # Moved from fit() method
            'eval_metric': 'rmse'  # Required for early stopping
        }

        self.params = params if params is not None else self.default_params
        self.models = {}  # Separate model for each target
        self.feature_importance = {}
        self.train_scores = {}
        self.val_scores = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.DataFrame] = None):
        """
        Fit separate XGBoost models for Y1 and Y2.

        Args:
            X_train: Training features
            y_train: Training targets (DataFrame with Y1 and Y2 columns)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Self for chaining
        """
        # Ensure we have Y1 and Y2 columns
        if not all(col in y_train.columns for col in ['Y1', 'Y2']):
            raise ValueError("y_train must contain columns 'Y1' and 'Y2'")

        for target in ['Y1', 'Y2']:
            print(f"\nTraining XGBoost for {target}...")

            # Prepare target data
            y_train_target = y_train[target].values
            y_val_target = y_val[target].values if y_val is not None else None

            # Create XGBoost model
            self.models[target] = xgb.XGBRegressor(**self.params)

            # Prepare evaluation sets
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train_target), (X_val, y_val_target)]
                eval_names = ['train', 'valid']
            else:
                eval_set = [(X_train, y_train_target)]
                eval_names = ['train']

            # Train model
            self.models[target].fit(
                X_train, y_train_target,
                eval_set=eval_set,
                verbose=False
            )

            # Store feature importance
            importance = self.models[target].feature_importances_
            self.feature_importance[target] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Calculate training scores
            train_pred = self.models[target].predict(X_train)
            self.train_scores[target] = {
                'r2': r2_score(y_train_target, train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train_target, train_pred))
            }

            # Calculate validation scores if validation set provided
            if X_val is not None and y_val is not None:
                val_pred = self.models[target].predict(X_val)
                self.val_scores[target] = {
                    'r2': r2_score(y_val_target, val_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val_target, val_pred))
                }

                # Check for overfitting
                train_val_gap = self.train_scores[target]['r2'] - self.val_scores[target]['r2']
                if train_val_gap > 0.05:
                    print(f"⚠️ Warning: Large train-val gap ({train_val_gap:.4f}) for {target}. "
                          f"Consider adding more regularization!")

                print(f"{target} - Train R²: {self.train_scores[target]['r2']:.4f}, "
                      f"Val R²: {self.val_scores[target]['r2']:.4f}, "
                      f"Best iter: {self.models[target].best_iteration}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for both Y1 and Y2.

        Args:
            X: Features DataFrame

        Returns:
            Array of shape (n_samples, 2) with Y1 and Y2 predictions
        """
        if not self.models:
            raise ValueError("Models not fitted yet! Call fit() first.")

        predictions = np.zeros((len(X), 2))

        for i, target in enumerate(['Y1', 'Y2']):
            if hasattr(self.models[target], 'best_iteration'):
                predictions[:, i] = self.models[target].predict(
                    X,
                    iteration_range=(0, self.models[target].best_iteration + 1)
                )
            else:
                predictions[:, i] = self.models[target].predict(X)

        return predictions

    def get_weighted_r2_score(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
        """
        Calculate weighted R² score for both targets.

        Args:
            y_true: True values (DataFrame with Y1 and Y2)
            y_pred: Predicted values (array with 2 columns)

        Returns:
            Weighted R² score
        """
        r2_y1 = r2_score(y_true['Y1'], y_pred[:, 0])
        r2_y2 = r2_score(y_true['Y2'], y_pred[:, 1])

        # Equal weighting for now, can be adjusted
        weighted_r2 = (r2_y1 + r2_y2) / 2

        return weighted_r2

    def get_feature_importance_summary(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get aggregated feature importance across both targets.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance summary
        """
        if not self.feature_importance:
            raise ValueError("No feature importance available. Fit the model first!")

        # Aggregate importance across targets
        importance_y1 = self.feature_importance['Y1'].set_index('feature')['importance']
        importance_y2 = self.feature_importance['Y2'].set_index('feature')['importance']

        avg_importance = (importance_y1 + importance_y2) / 2
        importance_df = pd.DataFrame({
            'feature': avg_importance.index,
            'avg_importance': avg_importance.values,
            'importance_y1': importance_y1.values,
            'importance_y2': importance_y2.values
        }).sort_values('avg_importance', ascending=False)

        # Add cumulative importance
        importance_df['cumulative_importance'] = importance_df['avg_importance'].cumsum()
        total_importance = importance_df['avg_importance'].sum()
        importance_df['cumulative_percent'] = importance_df['cumulative_importance'] / total_importance

        print(f"\nTop {min(top_n, len(importance_df))} features by average importance (XGBoost):")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['avg_importance']:.2f} "
                  f"(cumulative: {row['cumulative_percent']:.1%})")

        return importance_df.head(top_n)

    def compare_with_lgbm(self, lgbm_predictions: np.ndarray,
                          xgb_predictions: np.ndarray) -> Dict[str, float]:
        """
        Compare predictions with LightGBM to ensure diversity.

        Args:
            lgbm_predictions: Predictions from LightGBM model
            xgb_predictions: Predictions from XGBoost model

        Returns:
            Dictionary with correlation metrics
        """
        from scipy.stats import pearsonr

        # Calculate correlation for each target
        corr_y1, _ = pearsonr(lgbm_predictions[:, 0], xgb_predictions[:, 0])
        corr_y2, _ = pearsonr(lgbm_predictions[:, 1], xgb_predictions[:, 1])

        # Calculate overall correlation
        lgbm_flat = lgbm_predictions.flatten()
        xgb_flat = xgb_predictions.flatten()
        corr_overall, _ = pearsonr(lgbm_flat, xgb_flat)

        diversity_metrics = {
            'correlation_y1': corr_y1,
            'correlation_y2': corr_y2,
            'correlation_overall': corr_overall,
            'sufficient_diversity': corr_overall < 0.9
        }

        print("\nModel Diversity Analysis (XGBoost vs LightGBM):")
        print(f"  Y1 correlation: {corr_y1:.4f}")
        print(f"  Y2 correlation: {corr_y2:.4f}")
        print(f"  Overall correlation: {corr_overall:.4f}")

        if diversity_metrics['sufficient_diversity']:
            print("  ✓ Sufficient diversity for ensemble")
        else:
            print("  ⚠️ Warning: High correlation between models. Consider different hyperparameters.")

        return diversity_metrics

    def save_models(self, path_prefix: str):
        """
        Save trained models to disk.

        Args:
            path_prefix: Path prefix for saving models
        """
        for target, model in self.models.items():
            model_path = f"{path_prefix}_xgb_{target.lower()}.json"
            model.save_model(model_path)
            print(f"Saved {target} model to {model_path}")

    def load_models(self, path_prefix: str):
        """
        Load trained models from disk.

        Args:
            path_prefix: Path prefix for loading models
        """
        for target in ['Y1', 'Y2']:
            model_path = f"{path_prefix}_xgb_{target.lower()}.json"
            self.models[target] = xgb.XGBRegressor()
            self.models[target].load_model(model_path)
            print(f"Loaded {target} model from {model_path}")

    def cross_validate(self, X: pd.DataFrame, y: pd.DataFrame, cv) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Perform cross-validation and return scores.

        Args:
            X: Features DataFrame
            y: Targets DataFrame
            cv: Cross-validation splitter (e.g., PurgedTimeSeriesCV)

        Returns:
            Tuple of (cv_scores dictionary, out-of-fold predictions)
        """
        cv_scores = {'Y1': [], 'Y2': [], 'weighted': []}
        oof_predictions = np.zeros((len(X), 2))

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\n{'='*50}")
            print(f"XGBoost - Fold {fold_idx + 1}/{cv.get_n_splits()}")
            print(f"{'='*50}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            self.fit(X_train, y_train, X_val, y_val)

            # Generate predictions
            val_pred = self.predict(X_val)
            oof_predictions[val_idx] = val_pred

            # Calculate scores
            r2_y1 = r2_score(y_val['Y1'], val_pred[:, 0])
            r2_y2 = r2_score(y_val['Y2'], val_pred[:, 1])
            weighted_r2 = self.get_weighted_r2_score(y_val, val_pred)

            cv_scores['Y1'].append(r2_y1)
            cv_scores['Y2'].append(r2_y2)
            cv_scores['weighted'].append(weighted_r2)

            print(f"Fold {fold_idx + 1} Results:")
            print(f"  Y1 R²: {r2_y1:.4f}")
            print(f"  Y2 R²: {r2_y2:.4f}")
            print(f"  Weighted R²: {weighted_r2:.4f}")

        # Calculate average scores
        avg_scores = {
            'Y1_mean': np.mean(cv_scores['Y1']),
            'Y1_std': np.std(cv_scores['Y1']),
            'Y2_mean': np.mean(cv_scores['Y2']),
            'Y2_std': np.std(cv_scores['Y2']),
            'weighted_mean': np.mean(cv_scores['weighted']),
            'weighted_std': np.std(cv_scores['weighted'])
        }

        print(f"\n{'='*50}")
        print("XGBoost Cross-Validation Summary:")
        print(f"  Y1: {avg_scores['Y1_mean']:.4f} ± {avg_scores['Y1_std']:.4f}")
        print(f"  Y2: {avg_scores['Y2_mean']:.4f} ± {avg_scores['Y2_std']:.4f}")
        print(f"  Weighted: {avg_scores['weighted_mean']:.4f} ± {avg_scores['weighted_std']:.4f}")

        return avg_scores, oof_predictions