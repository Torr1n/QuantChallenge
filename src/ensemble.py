"""
Ensemble Methods for Model Combination

Implements middle averaging and weighted ensemble techniques proven in Jane Street competition.
Middle averaging shown to add 2-3% performance improvement.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class MiddleAveragingEnsemble:
    """
    Middle averaging ensemble that uses the middle portion of predictions.

    Critical insight from Jane Street: Using middle 50-60% of predictions
    reduces impact of outlier predictions and adds 2-3% performance.
    """

    def __init__(self, keep_ratio: float = 0.6):
        """
        Initialize middle averaging ensemble.

        Args:
            keep_ratio: Ratio of predictions to keep (0.6 = middle 60%)
        """
        self.keep_ratio = keep_ratio
        self.n_models = None

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate ensemble predictions using middle averaging.

        Args:
            predictions: Dictionary of model_name -> predictions array

        Returns:
            Ensemble predictions array
        """
        # Convert predictions dict to array
        model_names = list(predictions.keys())
        self.n_models = len(model_names)

        if self.n_models < 3:
            print("Warning: Middle averaging works best with 3+ models. Using simple average.")
            return self.simple_average(predictions)

        # Stack predictions: shape (n_models, n_samples, n_targets)
        pred_array = np.stack([predictions[name] for name in model_names])

        # Sort along model axis for each sample and target
        sorted_preds = np.sort(pred_array, axis=0)

        # Calculate indices for middle portion
        n_keep = int(self.n_models * self.keep_ratio)
        if n_keep < 1:
            n_keep = 1

        start_idx = (self.n_models - n_keep) // 2
        end_idx = start_idx + n_keep

        # Take middle predictions and average
        middle_preds = sorted_preds[start_idx:end_idx]
        ensemble_pred = np.mean(middle_preds, axis=0)

        print(f"Middle averaging: Using {n_keep}/{self.n_models} models "
              f"(indices {start_idx}:{end_idx})")

        return ensemble_pred

    def simple_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Simple averaging fallback for small ensembles.

        Args:
            predictions: Dictionary of model_name -> predictions array

        Returns:
            Simple average of predictions
        """
        pred_array = np.stack(list(predictions.values()))
        return np.mean(pred_array, axis=0)

    def optimize_keep_ratio(self, predictions: Dict[str, np.ndarray],
                           true_values: pd.DataFrame,
                           search_range: Tuple[float, float] = (0.4, 0.8)) -> float:
        """
        Optimize the keep_ratio parameter using validation data.

        Args:
            predictions: Dictionary of model_name -> predictions array
            true_values: True target values
            search_range: Range to search for optimal keep_ratio

        Returns:
            Optimal keep_ratio value
        """
        best_ratio = self.keep_ratio
        best_score = -float('inf')

        # Grid search over keep_ratio values
        for ratio in np.linspace(search_range[0], search_range[1], 21):
            self.keep_ratio = ratio
            ensemble_pred = self.predict(predictions)

            # Calculate weighted R² score
            r2_y1 = r2_score(true_values['Y1'], ensemble_pred[:, 0])
            r2_y2 = r2_score(true_values['Y2'], ensemble_pred[:, 1])
            weighted_r2 = (r2_y1 + r2_y2) / 2

            if weighted_r2 > best_score:
                best_score = weighted_r2
                best_ratio = ratio

        self.keep_ratio = best_ratio
        print(f"Optimal keep_ratio: {best_ratio:.2f} (R²: {best_score:.4f})")

        return best_ratio


class WeightedEnsemble:
    """
    Weighted ensemble with optimization using out-of-fold predictions.
    """

    def __init__(self, min_weight: float = 0.1, max_weight: float = 0.5):
        """
        Initialize weighted ensemble.

        Args:
            min_weight: Minimum weight for any model (prevents ignoring models)
            max_weight: Maximum weight for any model (prevents single model dominance)
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights = None
        self.model_names = None

    def optimize_weights(self, oof_predictions: Dict[str, np.ndarray],
                        true_values: pd.DataFrame,
                        method: str = 'differential_evolution') -> np.ndarray:
        """
        Optimize ensemble weights using out-of-fold predictions.

        Args:
            oof_predictions: Dictionary of model_name -> OOF predictions
            true_values: True target values
            method: Optimization method ('differential_evolution' or 'minimize')

        Returns:
            Optimal weights array
        """
        self.model_names = list(oof_predictions.keys())
        n_models = len(self.model_names)

        # Convert predictions to array
        pred_array = np.stack([oof_predictions[name] for name in self.model_names])

        def objective(weights):
            """Objective function to minimize (negative R²)."""
            # Normalize weights
            weights = weights / weights.sum()

            # Calculate weighted predictions
            weighted_pred = np.zeros_like(pred_array[0])
            for i, w in enumerate(weights):
                weighted_pred += w * pred_array[i]

            # Calculate weighted R² score
            r2_y1 = r2_score(true_values['Y1'], weighted_pred[:, 0])
            r2_y2 = r2_score(true_values['Y2'], weighted_pred[:, 1])
            weighted_r2 = (r2_y1 + r2_y2) / 2

            return -weighted_r2  # Minimize negative R²

        # Set bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]

        if method == 'differential_evolution':
            # Global optimization
            result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-7
            )
            self.weights = result.x / result.x.sum()

        else:
            # Local optimization with multiple random starts
            best_weights = None
            best_score = float('inf')

            for _ in range(10):
                # Random initial weights
                x0 = np.random.uniform(self.min_weight, self.max_weight, n_models)
                x0 = x0 / x0.sum()

                # Constraint: weights sum to 1
                constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )

                if result.fun < best_score:
                    best_score = result.fun
                    best_weights = result.x

            self.weights = best_weights / best_weights.sum()

        # Print optimized weights
        print("\nOptimized Ensemble Weights:")
        for name, weight in zip(self.model_names, self.weights):
            print(f"  {name}: {weight:.3f}")

        # Calculate final score
        final_score = -objective(self.weights)
        print(f"\nWeighted ensemble R²: {final_score:.4f}")

        return self.weights

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate weighted ensemble predictions.

        Args:
            predictions: Dictionary of model_name -> predictions array

        Returns:
            Weighted ensemble predictions
        """
        if self.weights is None:
            raise ValueError("Weights not optimized yet. Call optimize_weights first!")

        # Ensure same model order
        if set(predictions.keys()) != set(self.model_names):
            raise ValueError("Model names don't match trained ensemble!")

        # Calculate weighted predictions
        weighted_pred = np.zeros_like(list(predictions.values())[0])

        for name, weight in zip(self.model_names, self.weights):
            weighted_pred += weight * predictions[name]

        return weighted_pred


class StackingEnsemble:
    """
    Stacking ensemble with a meta-learner.
    """

    def __init__(self, meta_model_type: str = 'ridge'):
        """
        Initialize stacking ensemble.

        Args:
            meta_model_type: Type of meta-model ('ridge', 'linear', 'lightgbm')
        """
        self.meta_model_type = meta_model_type
        self.meta_models = {}  # Separate meta-model for each target
        self.model_names = None

    def fit(self, oof_predictions: Dict[str, np.ndarray],
            true_values: pd.DataFrame):
        """
        Fit meta-learner on out-of-fold predictions.

        Args:
            oof_predictions: Dictionary of model_name -> OOF predictions
            true_values: True target values
        """
        from sklearn.linear_model import RidgeCV, LinearRegression

        self.model_names = list(oof_predictions.keys())

        # Prepare meta-features (concatenate all predictions)
        meta_features = []
        for name in self.model_names:
            preds = oof_predictions[name]
            # Add both Y1 and Y2 predictions as features
            meta_features.append(preds[:, 0])  # Y1 predictions
            meta_features.append(preds[:, 1])  # Y2 predictions

        meta_features = np.column_stack(meta_features)

        # Train separate meta-models for Y1 and Y2
        for i, target in enumerate(['Y1', 'Y2']):
            print(f"\nTraining meta-model for {target}...")

            if self.meta_model_type == 'ridge':
                self.meta_models[target] = RidgeCV(
                    alphas=[0.01, 0.1, 1.0, 10.0],
                    cv=3
                )
            elif self.meta_model_type == 'linear':
                self.meta_models[target] = LinearRegression()
            elif self.meta_model_type == 'lightgbm':
                import lightgbm as lgb
                self.meta_models[target] = lgb.LGBMRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            else:
                raise ValueError(f"Unknown meta-model type: {self.meta_model_type}")

            # Fit meta-model
            self.meta_models[target].fit(meta_features, true_values[target])

            # Calculate meta-model performance
            meta_pred = self.meta_models[target].predict(meta_features)
            meta_r2 = r2_score(true_values[target], meta_pred)
            print(f"  Meta-model R² for {target}: {meta_r2:.4f}")

    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate stacking ensemble predictions.

        Args:
            predictions: Dictionary of model_name -> predictions array

        Returns:
            Stacking ensemble predictions
        """
        if not self.meta_models:
            raise ValueError("Meta-models not fitted yet. Call fit first!")

        # Prepare meta-features
        meta_features = []
        for name in self.model_names:
            preds = predictions[name]
            meta_features.append(preds[:, 0])  # Y1 predictions
            meta_features.append(preds[:, 1])  # Y2 predictions

        meta_features = np.column_stack(meta_features)

        # Generate predictions
        ensemble_pred = np.zeros((len(meta_features), 2))
        for i, target in enumerate(['Y1', 'Y2']):
            ensemble_pred[:, i] = self.meta_models[target].predict(meta_features)

        return ensemble_pred


def compare_ensemble_methods(predictions: Dict[str, np.ndarray],
                            true_values: pd.DataFrame) -> pd.DataFrame:
    """
    Compare different ensemble methods and return performance metrics.

    Args:
        predictions: Dictionary of model_name -> predictions array
        true_values: True target values

    Returns:
        DataFrame with comparison results
    """
    results = []

    # Simple average
    simple_avg = np.mean(list(predictions.values()), axis=0)
    r2_y1 = r2_score(true_values['Y1'], simple_avg[:, 0])
    r2_y2 = r2_score(true_values['Y2'], simple_avg[:, 1])
    results.append({
        'Method': 'Simple Average',
        'R2_Y1': r2_y1,
        'R2_Y2': r2_y2,
        'R2_Weighted': (r2_y1 + r2_y2) / 2
    })

    # Middle averaging
    for ratio in [0.5, 0.6, 0.7]:
        middle_ens = MiddleAveragingEnsemble(keep_ratio=ratio)
        middle_pred = middle_ens.predict(predictions)
        r2_y1 = r2_score(true_values['Y1'], middle_pred[:, 0])
        r2_y2 = r2_score(true_values['Y2'], middle_pred[:, 1])
        results.append({
            'Method': f'Middle Average ({int(ratio*100)}%)',
            'R2_Y1': r2_y1,
            'R2_Y2': r2_y2,
            'R2_Weighted': (r2_y1 + r2_y2) / 2
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).sort_values('R2_Weighted', ascending=False)

    print("\nEnsemble Methods Comparison:")
    print(comparison_df.to_string(index=False))

    # Calculate improvement over best single model
    single_model_scores = []
    for name, pred in predictions.items():
        r2_y1 = r2_score(true_values['Y1'], pred[:, 0])
        r2_y2 = r2_score(true_values['Y2'], pred[:, 1])
        single_model_scores.append((r2_y1 + r2_y2) / 2)

    best_single = max(single_model_scores)
    best_ensemble = comparison_df['R2_Weighted'].max()
    improvement = best_ensemble - best_single

    print(f"\nBest single model R²: {best_single:.4f}")
    print(f"Best ensemble R²: {best_ensemble:.4f}")
    print(f"Improvement: +{improvement:.4f} ({improvement/best_single*100:.1f}%)")

    return comparison_df