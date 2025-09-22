"""
Advanced Feature Engineering with Temporal Group Separation

Implements separate feature engineering for time-series and non-time-series groups
based on ACF analysis showing 96-unit seasonality in specific features.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Feature engineering with separate handling for temporal and non-temporal groups.

    Critical insight: Data contains two distinct groups:
    - Non-temporal: C, E, G, H, J, M, N -> Y1
    - Temporal (96-unit cycle): A, B, D, F, I, K, L -> Y2
    """

    # Feature groups based on ACF analysis
    NON_TEMPORAL_FEATURES = ['C', 'E', 'G', 'H', 'J', 'M', 'N']
    TEMPORAL_FEATURES = ['A', 'B', 'D', 'F', 'I', 'K', 'L']
    SEASONALITY_PERIOD = 96  # Critical: 96-unit seasonality discovered in data

    def __init__(self,
                 n_top_interactions: int = 35,
                 rolling_window: int = 96,  # Changed to match seasonality
                 min_correlation: float = 0.1):
        """
        Initialize AdvancedFeatureEngineer.

        Args:
            n_top_interactions: Number of top interaction features to select
            rolling_window: Window size for rolling statistics (96 for seasonality)
            min_correlation: Minimum correlation with target for feature selection
        """
        self.n_top_interactions = n_top_interactions
        self.rolling_window = rolling_window
        self.min_correlation = min_correlation

        self.selected_interactions = None
        self.scaler = None
        self.rolling_stats_buffer = {}  # Store last window of training data
        self.feature_stats = {}

    def select_targeted_interactions(self, X: pd.DataFrame, y: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Select interactions with awareness of feature groups.

        Prioritizes:
        - Within-group interactions for temporal features
        - Cross-group interactions for information transfer
        - Y1-correlated interactions from non-temporal features
        - Y2-correlated interactions from temporal features
        """
        print(f"Selecting targeted interaction features...")

        interactions = []
        scores = []

        # Within non-temporal group (for Y1)
        for col1 in self.NON_TEMPORAL_FEATURES:
            if col1 not in X.columns:
                continue
            for col2 in self.NON_TEMPORAL_FEATURES:
                if col2 not in X.columns or col1 >= col2:
                    continue

                interaction = X[col1] * X[col2]
                interaction_clean = interaction.fillna(0)

                # Focus on Y1 for non-temporal features
                mi_score = mutual_info_regression(
                    interaction_clean.values.reshape(-1, 1),
                    y['Y1'].fillna(0).values,
                    random_state=42
                )[0]

                interactions.append((col1, col2))
                scores.append(mi_score)

        # Within temporal group (for Y2)
        for col1 in self.TEMPORAL_FEATURES:
            if col1 not in X.columns:
                continue
            for col2 in self.TEMPORAL_FEATURES:
                if col2 not in X.columns or col1 >= col2:
                    continue

                interaction = X[col1] * X[col2]
                interaction_clean = interaction.fillna(0)

                # Focus on Y2 for temporal features
                mi_score = mutual_info_regression(
                    interaction_clean.values.reshape(-1, 1),
                    y['Y2'].fillna(0).values,
                    random_state=42
                )[0]

                interactions.append((col1, col2))
                scores.append(mi_score * 1.1)  # Slight boost for temporal interactions

        # Cross-group interactions (for both targets)
        for col1 in self.NON_TEMPORAL_FEATURES:
            if col1 not in X.columns:
                continue
            for col2 in self.TEMPORAL_FEATURES:
                if col2 not in X.columns:
                    continue

                interaction = X[col1] * X[col2]
                interaction_clean = interaction.fillna(0)

                # Average MI for both targets
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

                interactions.append((col1, col2))
                scores.append((mi_y1 + mi_y2) / 2)

        # Sort and select top N
        sorted_pairs = sorted(zip(scores, interactions), reverse=True)
        self.selected_interactions = [pair for _, pair in sorted_pairs[:self.n_top_interactions]]

        print(f"Top 5 interactions by MI score:")
        for i in range(min(5, len(sorted_pairs))):
            score, (col1, col2) = sorted_pairs[i]
            group1 = "T" if col1 in self.TEMPORAL_FEATURES else "N"
            group2 = "T" if col2 in self.TEMPORAL_FEATURES else "N"
            print(f"  {col1}({group1}) * {col2}({group2}): {score:.4f}")

        return self.selected_interactions

    def create_rolling_features_with_continuity(self, X: pd.DataFrame,
                                               is_train: bool = True,
                                               fold_idx: int = 0) -> pd.DataFrame:
        """
        Create rolling features with proper temporal continuity.

        For validation data, uses the last window of training data to maintain continuity.
        Only applies rolling features to temporal group.
        """
        rolling_features = pd.DataFrame(index=X.index)

        # Only create rolling features for temporal features
        for col in self.TEMPORAL_FEATURES:
            if col not in X.columns:
                continue

            if is_train:
                # Training: Calculate rolling statistics normally
                rolling_mean = X[col].rolling(window=self.rolling_window, min_periods=1).mean()
                rolling_std = X[col].rolling(window=self.rolling_window, min_periods=1).std()
                rolling_min = X[col].rolling(window=self.rolling_window, min_periods=1).min()
                rolling_max = X[col].rolling(window=self.rolling_window, min_periods=1).max()

                # Store last window for validation continuity
                self.rolling_stats_buffer[f'{col}_fold{fold_idx}'] = {
                    'last_window': X[col].iloc[-self.rolling_window:].values if len(X) >= self.rolling_window else X[col].values,
                    'last_mean': rolling_mean.iloc[-1],
                    'last_std': rolling_std.iloc[-1] if rolling_std.iloc[-1] == rolling_std.iloc[-1] else 1.0,
                    'last_min': rolling_min.iloc[-1],
                    'last_max': rolling_max.iloc[-1]
                }

                # Create 96-lag feature for temporal features (critical for seasonality)
                if len(X) > self.SEASONALITY_PERIOD:
                    rolling_features[f'{col}_lag96'] = X[col].shift(self.SEASONALITY_PERIOD).fillna(rolling_mean)
                else:
                    rolling_features[f'{col}_lag96'] = rolling_mean

            else:
                # Validation: Use stored buffer for continuity
                buffer_key = f'{col}_fold{fold_idx}'
                if buffer_key not in self.rolling_stats_buffer:
                    # Fallback if buffer not found
                    rolling_mean = pd.Series(0, index=X.index)
                    rolling_std = pd.Series(1, index=X.index)
                    rolling_min = pd.Series(0, index=X.index)
                    rolling_max = pd.Series(0, index=X.index)
                else:
                    buffer = self.rolling_stats_buffer[buffer_key]

                    # Initialize with training statistics
                    rolling_mean = pd.Series(buffer['last_mean'], index=X.index)
                    rolling_std = pd.Series(buffer['last_std'], index=X.index)
                    rolling_min = pd.Series(buffer['last_min'], index=X.index)
                    rolling_max = pd.Series(buffer['last_max'], index=X.index)

                    # For first few validation samples, use expanding window including training buffer
                    for i in range(min(self.rolling_window, len(X))):
                        # Combine last training window with current validation data
                        combined = np.concatenate([buffer['last_window'], X[col].iloc[:i+1].values])
                        if len(combined) > self.rolling_window:
                            combined = combined[-self.rolling_window:]

                        rolling_mean.iloc[i] = np.mean(combined)
                        rolling_std.iloc[i] = np.std(combined) if np.std(combined) > 0 else 1.0
                        rolling_min.iloc[i] = np.min(combined)
                        rolling_max.iloc[i] = np.max(combined)

                # No lag features for validation (would require future data)
                rolling_features[f'{col}_lag96'] = rolling_mean  # Use mean as proxy

            # Common features for both train and validation
            rolling_features[f'{col}_roll_mean'] = rolling_mean
            rolling_features[f'{col}_roll_std'] = rolling_std.fillna(1.0)
            rolling_features[f'{col}_roll_range'] = (rolling_max - rolling_min).fillna(0)
            rolling_features[f'{col}_roll_zscore'] = (
                (X[col] - rolling_mean) / (rolling_std + 1e-8)
            ).fillna(0)

            # Seasonality features (96-unit cycle)
            # Create for both train and validation to maintain feature consistency
            if len(X) > self.SEASONALITY_PERIOD:
                rolling_features[f'{col}_seasonal_diff'] = (
                    X[col] - X[col].shift(self.SEASONALITY_PERIOD)
                ).fillna(0)
            else:
                # For validation or small datasets, use 0 as placeholder
                rolling_features[f'{col}_seasonal_diff'] = 0

        # For non-temporal features, just add basic transformations
        for col in self.NON_TEMPORAL_FEATURES:
            if col not in X.columns:
                continue
            # Simple transformations that don't depend on time
            rolling_features[f'{col}_squared'] = X[col] ** 2
            rolling_features[f'{col}_abs'] = np.abs(X[col])

        return rolling_features

    def create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features based on selected pairs."""
        if self.selected_interactions is None:
            raise ValueError("Must call select_targeted_interactions first!")

        interaction_features = pd.DataFrame(index=X.index)

        for col1, col2 in self.selected_interactions:
            if col1 not in X.columns or col2 not in X.columns:
                continue
            interaction_features[f'{col1}_{col2}_interact'] = X[col1] * X[col2]

        return interaction_features

    def fit_transform(self, X: pd.DataFrame, y: pd.DataFrame, fold_idx: int = 0) -> pd.DataFrame:
        """
        Fit feature engineering on training data and transform.

        Args:
            X: Training features DataFrame
            y: Training targets DataFrame
            fold_idx: Current fold index for buffer management

        Returns:
            Transformed features DataFrame
        """
        print("Fitting advanced feature engineer on training data...")

        # Select targeted interactions
        self.select_targeted_interactions(X, y)

        # Create all features
        interaction_feats = self.create_interaction_features(X)
        rolling_feats = self.create_rolling_features_with_continuity(X, is_train=True, fold_idx=fold_idx)

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
        print(f"  Rolling/Temporal: {rolling_feats.shape[1]}")
        print(f"  Temporal features with rolling: {[c for c in self.TEMPORAL_FEATURES if c in X.columns]}")

        return all_features_scaled

    def transform(self, X: pd.DataFrame, fold_idx: int = 0) -> pd.DataFrame:
        """
        Transform validation/test data using fitted parameters.

        Args:
            X: Validation/test features DataFrame
            fold_idx: Current fold index for buffer management

        Returns:
            Transformed features DataFrame
        """
        if self.selected_interactions is None or self.scaler is None:
            raise ValueError("Must call fit_transform on training data first!")

        # Create features using fitted parameters
        interaction_feats = self.create_interaction_features(X)
        rolling_feats = self.create_rolling_features_with_continuity(X, is_train=False, fold_idx=fold_idx)

        # Combine all features
        all_features = pd.concat([X, interaction_feats, rolling_feats], axis=1)

        # Apply fitted scaler
        all_features_scaled = pd.DataFrame(
            self.scaler.transform(all_features.fillna(0)),
            index=all_features.index,
            columns=all_features.columns
        )

        return all_features_scaled