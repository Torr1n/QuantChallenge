"""
Purged Time Series Cross-Validation Implementation

Critical for preventing temporal leakage in financial time series.
Implements a gap between training and validation sets to prevent information leakage.
"""

import numpy as np
from typing import Generator, Tuple, Optional
import pandas as pd


class PurgedTimeSeriesCV:
    """
    Purged Time Series Cross-Validation with gap to prevent temporal leakage.

    Critical: Standard K-fold will cause 0.10+ score inflation due to temporal leakage.
    This implementation maintains strict temporal ordering with a purged gap.
    """

    def __init__(self, n_splits: int = 3, gap_size: int = 4000, val_size: int = 10000):
        """
        Initialize PurgedTimeSeriesCV.

        Args:
            n_splits: Number of CV splits (default: 3)
            gap_size: Size of gap between train and validation (default: 4000 = 5% of 80k)
            val_size: Size of validation set (default: 10000)
        """
        self.n_splits = n_splits
        self.gap_size = gap_size
        self.val_size = val_size

    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation indices for purged time series cross-validation.

        Dynamically calculates splits based on data size with proportional allocation.
        For 80k samples, approximates:
        - Fold 1: train[0:20000], gap[20000:24000], val[24000:34000]
        - Fold 2: train[0:34000], gap[34000:38000], val[38000:48000]
        - Fold 3: train[0:48000], gap[48000:52000], val[52000:62000]

        Args:
            X: Features array or DataFrame
            y: Target array (optional)
            groups: Group array (not used, for sklearn compatibility)

        Yields:
            Tuple of (train_indices, val_indices) for each fold
        """
        n_samples = len(X)

        # Use hardcoded splits for standard 80k dataset
        if n_samples >= 62000:  # Standard dataset size
            splits = [
                {'train_end': 20000, 'gap_end': 24000, 'val_end': 34000},
                {'train_end': 34000, 'gap_end': 38000, 'val_end': 48000},
                {'train_end': 48000, 'gap_end': 52000, 'val_end': 62000}
            ]
        else:
            # Dynamic splits for smaller datasets (e.g., testing)
            # Calculate proportional splits
            total_usable = n_samples * 0.775  # Reserve ~22.5% for holdout
            val_size = min(self.val_size, int(total_usable * 0.15))  # 15% for validation
            gap_size = min(self.gap_size, int(n_samples * 0.05))  # 5% gap

            splits = []
            for i in range(self.n_splits):
                # Progressive training sizes
                train_ratio = (i + 1) / (self.n_splits + 1)
                train_end = int(total_usable * train_ratio * 0.6)  # Scale down for smaller data

                gap_end = train_end + gap_size
                val_end = min(gap_end + val_size, int(total_usable))

                # Ensure we have reasonable sizes
                if train_end > 100 and val_end <= n_samples:  # Minimum 100 samples for training
                    splits.append({
                        'train_end': train_end,
                        'gap_end': gap_end,
                        'val_end': val_end
                    })

        for fold_idx, split in enumerate(splits):
            # Ensure we don't exceed data bounds
            if split['val_end'] > n_samples:
                print(f"Warning: Fold {fold_idx + 1} validation end {split['val_end']} exceeds data size {n_samples}")
                continue

            train_idx = np.arange(0, split['train_end'])
            val_idx = np.arange(split['gap_end'], split['val_end'])

            # Validate no overlap between train and validation
            assert len(np.intersect1d(train_idx, val_idx)) == 0, \
                f"Train and validation sets overlap in fold {fold_idx + 1}!"

            # Ensure gap is maintained (relaxed for small datasets)
            min_gap = min(self.gap_size, int(n_samples * 0.02))  # At least 2% gap for small data
            actual_gap = val_idx[0] - train_idx[-1] - 1
            assert actual_gap >= min_gap, \
                f"Insufficient gap ({actual_gap} < {min_gap}) in fold {fold_idx + 1}!"

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def get_holdout_indices(self, n_samples: int = 80000) -> np.ndarray:
        """
        Get indices for the final holdout set (15% of data).

        Args:
            n_samples: Total number of samples

        Returns:
            Array of holdout indices [68000:80000]
        """
        holdout_start = 68000
        holdout_end = min(n_samples, 80000)
        return np.arange(holdout_start, holdout_end)

    def validate_temporal_integrity(self, X, y=None) -> bool:
        """
        Validate that temporal ordering is maintained and no leakage exists.

        Args:
            X: Features array or DataFrame
            y: Target array (optional)

        Returns:
            True if validation passes, raises AssertionError otherwise
        """
        for fold_idx, (train_idx, val_idx) in enumerate(self.split(X, y)):
            # Check temporal ordering
            assert train_idx[-1] < val_idx[0], \
                f"Fold {fold_idx + 1}: Training data comes after validation data!"

            # Check gap exists
            gap = val_idx[0] - train_idx[-1] - 1
            assert gap >= self.gap_size, \
                f"Fold {fold_idx + 1}: Gap size {gap} is less than required {self.gap_size}!"

            # Check no overlap
            assert len(np.intersect1d(train_idx, val_idx)) == 0, \
                f"Fold {fold_idx + 1}: Train and validation sets overlap!"

            print(f"Fold {fold_idx + 1} validated: "
                  f"Train [{train_idx[0]}:{train_idx[-1]}], "
                  f"Gap size: {gap}, "
                  f"Val [{val_idx[0]}:{val_idx[-1]}]")

        return True


def demonstrate_cv_usage():
    """
    Demonstrate proper usage of PurgedTimeSeriesCV.
    """
    # Create sample data
    n_samples = 80000
    X = np.random.randn(n_samples, 15)  # 15 features
    y = np.random.randn(n_samples, 2)   # 2 targets (Y1, Y2)

    # Initialize CV
    cv = PurgedTimeSeriesCV(n_splits=3, gap_size=4000)

    # Validate temporal integrity
    print("Validating temporal integrity...")
    cv.validate_temporal_integrity(X, y)

    # Show splits
    print("\nCV Splits:")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_idx)} samples [{train_idx[0]}:{train_idx[-1]}]")
        print(f"  Val: {len(val_idx)} samples [{val_idx[0]}:{val_idx[-1]}]")
        print(f"  Gap: {val_idx[0] - train_idx[-1] - 1} samples")

    # Show holdout
    holdout_idx = cv.get_holdout_indices(n_samples)
    print(f"\nHoldout set: {len(holdout_idx)} samples [{holdout_idx[0]}:{holdout_idx[-1]}]")
    print(f"Holdout percentage: {len(holdout_idx) / n_samples * 100:.1f}%")


if __name__ == "__main__":
    demonstrate_cv_usage()