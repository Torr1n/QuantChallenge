"""
Complete Training Pipeline with Ensemble

Orchestrates the entire training process from data loading through ensemble optimization.
Follows the 48-hour competition timeline with go/no-go gates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.config import *
from src.utils.cross_validation import PurgedTimeSeriesCV
from src.features.feature_engineer import FeatureEngineer
from src.models.lgbm_model import LightGBMModel
from src.models.xgb_model import XGBoostModel
from src.models.neural_net import NeuralNetModel
from src.ensemble import MiddleAveragingEnsemble, WeightedEnsemble, compare_ensemble_methods


class CompetitionPipeline:
    """
    Complete pipeline for the quantitative competition.
    Implements all phases with proper validation and go/no-go gates.
    """

    def __init__(self):
        """Initialize the competition pipeline."""
        self.cv = PurgedTimeSeriesCV(**CV_PARAMS)
        self.fe = None
        self.models = {}
        self.oof_predictions = {}
        self.cv_scores = {}
        self.ensemble_models = {}
        self.start_time = time.time()

    def log_progress(self, message: str, level: str = "INFO"):
        """
        Log progress with timestamp.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, SUCCESS, ERROR)
        """
        elapsed = (time.time() - self.start_time) / 3600
        symbols = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "SUCCESS": "✅",
            "ERROR": "❌"
        }
        print(f"\n[{elapsed:.1f}h] {symbols.get(level, '')} {message}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate data."""
        self.log_progress("Loading data...", "INFO")

        df = pd.read_csv(TRAIN_FILE)
        X = df[FEATURE_COLS]
        y = df[TARGET_COLS]

        self.log_progress(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features", "SUCCESS")

        # Check target correlation
        target_corr = y.corr().iloc[0, 1]
        self.log_progress(f"Y1-Y2 correlation: {target_corr:.4f}", "INFO")

        return X, y

    def train_baseline_lgbm(self, X: pd.DataFrame, y: pd.DataFrame) -> bool:
        """
        Train LightGBM baseline model.

        Returns:
            True if CV score meets threshold, False otherwise
        """
        self.log_progress("Phase 1: Training LightGBM baseline", "INFO")

        oof_predictions = np.zeros((len(X), 2))
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            self.log_progress(f"LightGBM Fold {fold_idx + 1}/{self.cv.get_n_splits()}", "INFO")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Feature engineering
            if fold_idx == 0:
                self.fe = FeatureEngineer(**FE_PARAMS)
                X_train_fe = self.fe.fit_transform(X_train, y_train)
            else:
                # Re-fit for each fold
                fe_fold = FeatureEngineer(**FE_PARAMS)
                X_train_fe = fe_fold.fit_transform(X_train, y_train)
                X_val_fe = fe_fold.transform(X_val)

                # Use first fold's feature engineer for consistency
                if fold_idx == 0:
                    self.fe = fe_fold

            if fold_idx == 0:
                X_val_fe = self.fe.transform(X_val)

            # Train model
            model = LightGBMModel(params=LGBM_PARAMS)
            model.fit(X_train_fe, y_train, X_val_fe, y_val)

            # Predictions
            val_pred = model.predict(X_val_fe)
            oof_predictions[val_idx] = val_pred

            # Scores
            from sklearn.metrics import r2_score
            weighted_r2 = model.get_weighted_r2_score(y_val, val_pred)
            cv_scores.append(weighted_r2)

            self.log_progress(f"Fold {fold_idx + 1} R²: {weighted_r2:.4f}", "INFO")

        # Store results
        self.models['lgbm'] = model
        self.oof_predictions['lgbm'] = oof_predictions
        avg_cv = np.mean(cv_scores)
        self.cv_scores['lgbm'] = {
            'mean': avg_cv,
            'std': np.std(cv_scores),
            'folds': cv_scores
        }

        # Go/No-Go decision
        if avg_cv >= MIN_CV_SCORE:
            self.log_progress(f"LightGBM baseline PASSED: CV={avg_cv:.4f} >= {MIN_CV_SCORE}", "SUCCESS")
            return True
        else:
            self.log_progress(f"LightGBM baseline FAILED: CV={avg_cv:.4f} < {MIN_CV_SCORE}", "ERROR")
            return False

    def train_xgboost(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train XGBoost for ensemble diversity."""
        self.log_progress("Phase 2: Training XGBoost for diversity", "INFO")

        oof_predictions = np.zeros((len(X), 2))
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            self.log_progress(f"XGBoost Fold {fold_idx + 1}/{self.cv.get_n_splits()}", "INFO")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Feature engineering
            fe_fold = FeatureEngineer(**FE_PARAMS)
            X_train_fe = fe_fold.fit_transform(X_train, y_train)
            X_val_fe = fe_fold.transform(X_val)

            # Train model
            model = XGBoostModel(params=XGB_PARAMS)
            model.fit(X_train_fe, y_train, X_val_fe, y_val)

            # Predictions
            val_pred = model.predict(X_val_fe)
            oof_predictions[val_idx] = val_pred

            # Scores
            weighted_r2 = model.get_weighted_r2_score(y_val, val_pred)
            cv_scores.append(weighted_r2)

        # Store results
        self.models['xgboost'] = model
        self.oof_predictions['xgboost'] = oof_predictions
        avg_cv = np.mean(cv_scores)
        self.cv_scores['xgboost'] = {
            'mean': avg_cv,
            'std': np.std(cv_scores),
            'folds': cv_scores
        }

        # Check diversity
        lgbm_pred = self.oof_predictions['lgbm']
        xgb_pred = self.oof_predictions['xgboost']
        correlation = np.corrcoef(lgbm_pred.flatten(), xgb_pred.flatten())[0, 1]

        self.log_progress(f"XGBoost CV: {avg_cv:.4f}, Correlation with LightGBM: {correlation:.3f}", "INFO")

        if correlation < 0.9:
            self.log_progress("Sufficient model diversity achieved", "SUCCESS")
        else:
            self.log_progress("High correlation between models", "WARNING")

    def train_neural_network(self, X: pd.DataFrame, y: pd.DataFrame, time_remaining: float):
        """
        Train neural network if time permits.

        Args:
            time_remaining: Hours remaining in competition
        """
        if time_remaining < 6:
            self.log_progress("Skipping neural network due to time constraints", "WARNING")
            return

        self.log_progress("Phase 3: Training Neural Network", "INFO")

        oof_predictions = np.zeros((len(X), 2))
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            self.log_progress(f"Neural Net Fold {fold_idx + 1}/{self.cv.get_n_splits()}", "INFO")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Feature engineering
            fe_fold = FeatureEngineer(**FE_PARAMS)
            X_train_fe = fe_fold.fit_transform(X_train, y_train)
            X_val_fe = fe_fold.transform(X_val)

            # Train model
            model = NeuralNetModel(params=NN_PARAMS)
            model.fit(X_train_fe, y_train, X_val_fe, y_val)

            # Predictions
            val_pred = model.predict(X_val_fe)
            oof_predictions[val_idx] = val_pred

            # Scores
            weighted_r2 = model.get_weighted_r2_score(y_val, val_pred)
            cv_scores.append(weighted_r2)

        # Store results
        self.models['neural_net'] = model
        self.oof_predictions['neural_net'] = oof_predictions
        avg_cv = np.mean(cv_scores)
        self.cv_scores['neural_net'] = {
            'mean': avg_cv,
            'std': np.std(cv_scores),
            'folds': cv_scores
        }

        self.log_progress(f"Neural Network CV: {avg_cv:.4f}", "INFO")

    def create_ensemble(self, y: pd.DataFrame) -> Dict[str, float]:
        """
        Create and optimize ensemble models.

        Args:
            y: True target values

        Returns:
            Dictionary with ensemble scores
        """
        self.log_progress("Phase 4: Creating Ensemble", "INFO")

        # Compare ensemble methods
        comparison_df = compare_ensemble_methods(self.oof_predictions, y)

        # Middle averaging ensemble
        self.log_progress("Optimizing middle averaging ensemble", "INFO")
        middle_ens = MiddleAveragingEnsemble()
        optimal_ratio = middle_ens.optimize_keep_ratio(self.oof_predictions, y)
        self.ensemble_models['middle_averaging'] = middle_ens

        # Weighted ensemble
        self.log_progress("Optimizing weighted ensemble", "INFO")
        weighted_ens = WeightedEnsemble()
        weights = weighted_ens.optimize_weights(self.oof_predictions, y)
        self.ensemble_models['weighted'] = weighted_ens

        # Calculate final ensemble score
        final_pred = middle_ens.predict(self.oof_predictions)
        from sklearn.metrics import r2_score
        r2_y1 = r2_score(y['Y1'], final_pred[:, 0])
        r2_y2 = r2_score(y['Y2'], final_pred[:, 1])
        weighted_r2 = (r2_y1 + r2_y2) / 2

        ensemble_scores = {
            'r2_y1': r2_y1,
            'r2_y2': r2_y2,
            'weighted_r2': weighted_r2,
            'optimal_ratio': optimal_ratio
        }

        # Check improvement
        best_single = max([scores['mean'] for scores in self.cv_scores.values()])
        improvement = weighted_r2 - best_single

        if improvement > 0.02:
            self.log_progress(f"Ensemble improvement: +{improvement:.4f} (PASSED)", "SUCCESS")
        else:
            self.log_progress(f"Ensemble improvement: +{improvement:.4f} (minimal)", "WARNING")

        return ensemble_scores

    def final_validation(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Perform final validation on holdout set.

        Args:
            X: All features
            y: All targets
        """
        self.log_progress("Phase 5: Final Holdout Validation", "INFO")

        # Get holdout indices
        holdout_idx = self.cv.get_holdout_indices(len(X))
        train_idx = np.arange(0, holdout_idx[0])

        # Split data
        X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
        y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]

        self.log_progress(f"Training on {len(X_train)} samples, validating on {len(X_holdout)}", "INFO")

        # Train final models and get predictions
        holdout_predictions = {}

        # Re-fit feature engineer on all training data
        fe_final = FeatureEngineer(**FE_PARAMS)
        X_train_fe = fe_final.fit_transform(X_train, y_train)
        X_holdout_fe = fe_final.transform(X_holdout)

        # Train each model type
        for model_name in self.models.keys():
            self.log_progress(f"Training final {model_name}", "INFO")

            if model_name == 'lgbm':
                model = LightGBMModel(params=LGBM_PARAMS)
            elif model_name == 'xgboost':
                model = XGBoostModel(params=XGB_PARAMS)
            elif model_name == 'neural_net':
                model = NeuralNetModel(params=NN_PARAMS)

            model.fit(X_train_fe, y_train)
            holdout_predictions[model_name] = model.predict(X_holdout_fe)

        # Ensemble predictions
        ensemble_pred = self.ensemble_models['middle_averaging'].predict(holdout_predictions)

        # Calculate scores
        from sklearn.metrics import r2_score
        r2_y1 = r2_score(y_holdout['Y1'], ensemble_pred[:, 0])
        r2_y2 = r2_score(y_holdout['Y2'], ensemble_pred[:, 1])
        holdout_r2 = (r2_y1 + r2_y2) / 2

        self.log_progress(f"Holdout R²: {holdout_r2:.4f} (Y1: {r2_y1:.4f}, Y2: {r2_y2:.4f})", "INFO")

        # Compare with CV score
        cv_ensemble_score = max([scores['mean'] for scores in self.cv_scores.values()])
        gap = cv_ensemble_score - holdout_r2

        if gap < 0.03:
            self.log_progress(f"CV-Holdout gap: {gap:.4f} (ACCEPTABLE)", "SUCCESS")
        else:
            self.log_progress(f"CV-Holdout gap: {gap:.4f} (possible overfitting)", "WARNING")

    def save_artifacts(self):
        """Save all models and results."""
        self.log_progress("Saving artifacts", "INFO")

        # Save results summary
        results = {
            'cv_scores': self.cv_scores,
            'model_names': list(self.models.keys()),
            'elapsed_time': (time.time() - self.start_time) / 3600
        }

        with open(RESULTS_DIR / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save OOF predictions
        for name, preds in self.oof_predictions.items():
            np.save(RESULTS_DIR / f'oof_{name}.npy', preds)

        # Save ensemble models
        with open(MODELS_DIR / 'ensemble_models.pkl', 'wb') as f:
            pickle.dump(self.ensemble_models, f)

        self.log_progress("All artifacts saved", "SUCCESS")

    def run_full_pipeline(self):
        """
        Run the complete competition pipeline.
        """
        print("=" * 60)
        print("QUANTITATIVE COMPETITION PIPELINE")
        print("48-Hour Timeline Started")
        print("=" * 60)

        # Load data
        X, y = self.load_data()

        # Phase 1: Baseline (Hours 0-12)
        baseline_success = self.train_baseline_lgbm(X, y)

        if not baseline_success:
            self.log_progress("Pipeline stopped: Baseline performance insufficient", "ERROR")
            return

        # Phase 2: XGBoost (Hours 12-18)
        self.train_xgboost(X, y)

        # Check time remaining
        elapsed = (time.time() - self.start_time) / 3600
        time_remaining = 48 - elapsed

        # Phase 3: Neural Network (Hours 18-24, if time permits)
        self.train_neural_network(X, y, time_remaining)

        # Phase 4: Ensemble (Hours 24-36)
        ensemble_scores = self.create_ensemble(y)

        # Phase 5: Final Validation (Hours 36-42)
        self.final_validation(X, y)

        # Save everything
        self.save_artifacts()

        # Final summary
        elapsed_total = (time.time() - self.start_time) / 3600
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed_total:.1f} hours")
        print(f"Models trained: {list(self.models.keys())}")
        print(f"Best CV score: {max([s['mean'] for s in self.cv_scores.values()]):.4f}")
        print(f"Final ensemble R²: {ensemble_scores['weighted_r2']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = CompetitionPipeline()
    pipeline.run_full_pipeline()