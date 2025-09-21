"""
Feature Engineering Pipeline

Creates interaction features, rolling statistics, and transformations
while maintaining temporal integrity and preventing leakage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering pipeline with mutual information-based selection.

    Critical: All statistics must be calculated ONLY on training data to prevent leakage.
    """

    def __init__(self,
                 n_top_interactions: int = 35,
                 rolling_window: int = 150,
                 min_correlation: float = 0.1):
        """
        Initialize FeatureEngineer.

        Args:
            n_top_interactions: Number of top interaction features to select (default: 35)
            rolling_window: Window size for rolling statistics (default: 150)
            min_correlation: Minimum correlation with target for feature selection (default: 0.1)
        """
        self.n_top_interactions = n_top_interactions
        self.rolling_window = rolling_window
        self.min_correlation = min_correlation

        self.selected_interactions = None
        self.scaler = None
        self.feature_stats = {}

    def select_top_interactions(self, X: pd.DataFrame, y: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Select top interaction features using mutual information.

        Args:
            X: Features DataFrame with columns A-N
            y: Target DataFrame with Y1 and Y2

        Returns:
            List of tuples (col1, col2) representing selected interactions
        """
        print(f"Selecting top {self.n_top_interactions} interaction features...")

        feature_cols = X.columns.tolist()
        interactions = []
        scores = []

        # Calculate mutual information for all pairwise interactions
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i:]:  # Include self-interactions
                # Create interaction
                interaction = X[col1] * X[col2]
                interaction_clean = interaction.fillna(0)

                # Calculate mutual information with both targets
                mi_y1 = mutual_info_regression(
                    interaction_clean.values.reshape(-1, 1),
                    y['Y1'].fillna(0).values,
                    random_state=42
                )[0]

                mi_y2 = mutual_info_regression(
                    interaction_clean.values.reshape(-1, 1),
                    y['Y2'].fillna(0).values,
                    random_state=42
                )[0]

                # Average MI score across both targets
                avg_mi = (mi_y1 + mi_y2) / 2

                interactions.append((col1, col2))
                scores.append(avg_mi)

        # Sort by scores and select top N
        sorted_pairs = sorted(zip(scores, interactions), reverse=True)
        self.selected_interactions = [pair for _, pair in sorted_pairs[:self.n_top_interactions]]

        print(f"Top 5 interactions by MI score:")
        for i in range(min(5, len(sorted_pairs))):
            score, (col1, col2) = sorted_pairs[i]
            print(f"  {col1} * {col2}: {score:.4f}")

        return self.selected_interactions

    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features based on selected pairs.

        Args:
            X: Features DataFrame

        Returns:
            DataFrame with interaction features
        """
        if self.selected_interactions is None:
            raise ValueError("Must call select_top_interactions first!")

        interaction_features = pd.DataFrame(index=X.index)

        for col1, col2 in self.selected_interactions:
            if col1 == col2:
                # Self-interaction (squared feature)
                interaction_features[f'{col1}_squared'] = X[col1] ** 2
            else:
                interaction_features[f'{col1}_{col2}_interact'] = X[col1] * X[col2]

        return interaction_features

    def create_rolling_features(self, X: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Create rolling statistics features.

        Critical: Calculate statistics ONLY on training data to prevent leakage.

        Args:
            X: Features DataFrame
            is_train: Whether this is training data

        Returns:
            DataFrame with rolling features
        """
        rolling_features = pd.DataFrame(index=X.index)

        for col in X.columns:
            if is_train:
                # Calculate and store rolling statistics
                rolling_mean = X[col].rolling(window=self.rolling_window, min_periods=1).mean()
                rolling_std = X[col].rolling(window=self.rolling_window, min_periods=1).std()
                rolling_min = X[col].rolling(window=self.rolling_window, min_periods=1).min()
                rolling_max = X[col].rolling(window=self.rolling_window, min_periods=1).max()

                # Store parameters for validation set
                self.feature_stats[col] = {
                    'last_mean': rolling_mean.iloc[-1],
                    'last_std': rolling_std.iloc[-1],
                    'last_min': rolling_min.iloc[-1],
                    'last_max': rolling_max.iloc[-1]
                }
            else:
                # Use stored statistics from training
                if col not in self.feature_stats:
                    raise ValueError(f"Statistics not found for feature {col}. Fit on training data first!")

                # Apply forward-fill with training statistics
                rolling_mean = pd.Series(self.feature_stats[col]['last_mean'], index=X.index)
                rolling_std = pd.Series(self.feature_stats[col]['last_std'], index=X.index)
                rolling_min = pd.Series(self.feature_stats[col]['last_min'], index=X.index)
                rolling_max = pd.Series(self.feature_stats[col]['last_max'], index=X.index)

            # Create rolling features
            rolling_features[f'{col}_roll_mean'] = rolling_mean
            rolling_features[f'{col}_roll_std'] = rolling_std.fillna(0)
            rolling_features[f'{col}_roll_range'] = (rolling_max - rolling_min).fillna(0)

            # Relative position within rolling window
            if is_train:
                rolling_features[f'{col}_roll_zscore'] = (
                    (X[col] - rolling_mean) / (rolling_std + 1e-8)
                ).fillna(0)

        return rolling_features

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature engineering on training data and transform.

        Args:
            X: Training features DataFrame
            y: Training targets DataFrame

        Returns:
            Transformed features DataFrame
        """
        print("Fitting feature engineer on training data...")

        # Select interaction features
        self.select_top_interactions(X, y)

        # Create all features
        interaction_feats = self.create_interaction_features(X)
        rolling_feats = self.create_rolling_features(X, is_train=True)

        # Combine all features
        all_features = pd.concat([X, interaction_feats, rolling_feats], axis=1)

        # Fit scaler on combined features
        self.scaler = StandardScaler()
        all_features_scaled = pd.DataFrame(
            self.scaler.fit_transform(all_features.fillna(0)),
            index=all_features.index,
            columns=all_features.columns
        )

        print(f"Total features created: {all_features_scaled.shape[1]}")
        print(f"  Original: {X.shape[1]}")
        print(f"  Interactions: {interaction_feats.shape[1]}")
        print(f"  Rolling: {rolling_feats.shape[1]}")

        return all_features_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform validation/test data using fitted parameters.

        Args:
            X: Validation/test features DataFrame

        Returns:
            Transformed features DataFrame
        """
        if self.selected_interactions is None or self.scaler is None:
            raise ValueError("Must call fit_transform on training data first!")

        # Create features using fitted parameters
        interaction_feats = self.create_interaction_features(X)
        rolling_feats = self.create_rolling_features(X, is_train=False)

        # Combine all features
        all_features = pd.concat([X, interaction_feats, rolling_feats], axis=1)

        # Apply fitted scaler
        all_features_scaled = pd.DataFrame(
            self.scaler.transform(all_features.fillna(0)),
            index=all_features.index,
            columns=all_features.columns
        )

        return all_features_scaled

    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from a fitted model.

        Args:
            model: Fitted model with feature_importances_ attribute
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance rankings
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Calculate cumulative importance
            importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
            importance_df['cumulative_percent'] = importance_df['cumulative_importance'] / importance_df['importance'].sum()

            return importance_df
        else:
            return pd.DataFrame()


def validate_no_leakage(train_features: pd.DataFrame,
                        val_features: pd.DataFrame,
                        train_idx: np.ndarray,
                        val_idx: np.ndarray) -> bool:
    """
    Validate that no information leakage exists between train and validation sets.

    Args:
        train_features: Training features
        val_features: Validation features
        train_idx: Training indices
        val_idx: Validation indices

    Returns:
        True if no leakage detected
    """
    # Check temporal ordering
    assert train_idx[-1] < val_idx[0], "Training indices come after validation indices!"

    # Check for NaN patterns that might indicate leakage
    train_nan_pattern = train_features.isna().sum()
    val_nan_pattern = val_features.isna().sum()

    # Rolling features should have different patterns
    rolling_cols = [col for col in train_features.columns if 'roll' in col]
    if len(rolling_cols) > 0:
        for col in rolling_cols[:5]:  # Check first 5 rolling features
            train_vals = train_features[col].dropna().values[-10:]
            val_vals = val_features[col].dropna().values[:10]

            # Values should be different (no exact matches)
            if len(train_vals) > 0 and len(val_vals) > 0:
                assert not np.array_equal(train_vals, val_vals), \
                    f"Potential leakage in {col}: identical values between train and val"

    print("✓ No temporal leakage detected")
    return True