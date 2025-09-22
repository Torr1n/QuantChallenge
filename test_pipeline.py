"""
Minimal Testing Pipeline with Observability

Lightweight version for testing on laptop with reduced computational requirements.
Includes comprehensive observability and validation checks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure for minimal testing
TEST_CONFIG = {
    'sample_size': 10000,  # Use only 10k samples for testing
    'n_folds': 2,          # Reduced from 3 folds
    'n_estimators': 50,    # Reduced from 300 trees
    'n_interactions': 10,  # Reduced from 35 interactions
    'nn_epochs': 10,       # Reduced from 100 epochs
    'quick_mode': True
}

class MinimalTestPipeline:
    """
    Minimal pipeline for testing and observability on limited hardware.
    """

    def __init__(self, test_mode=True):
        """Initialize test pipeline with reduced settings."""
        self.test_mode = test_mode
        self.results = {}
        self.timings = {}
        self.validation_checks = {}

        # Override config for testing
        if test_mode:
            self._setup_test_config()

    def _setup_test_config(self):
        """Configure minimal settings for testing."""
        print("=" * 60)
        print("MINIMAL TEST MODE ACTIVATED")
        print("=" * 60)
        print(f"Sample size: {TEST_CONFIG['sample_size']}")
        print(f"Folds: {TEST_CONFIG['n_folds']}")
        print(f"Trees: {TEST_CONFIG['n_estimators']}")
        print(f"Interactions: {TEST_CONFIG['n_interactions']}")
        print("=" * 60 + "\n")

    def load_test_data(self):
        """Load a subset of data for testing."""
        start_time = time.time()

        print("📊 Loading test data subset...")
        df = pd.read_csv('data/train.csv', nrows=TEST_CONFIG['sample_size'])

        feature_cols = [chr(ord('A') + i) for i in range(14)]
        target_cols = ['Y1', 'Y2']

        X = df[feature_cols]
        y = df[target_cols]

        load_time = time.time() - start_time
        self.timings['data_load'] = load_time

        print(f"✓ Loaded {len(X)} samples in {load_time:.2f}s")
        print(f"  Features: {X.shape[1]}, Targets: {y.shape[1]}")
        print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

        return X, y

    def validate_cv_setup(self, X, y):
        """Test and validate cross-validation setup."""
        print("\n🔍 Validating Cross-Validation Setup...")

        from src.utils.cross_validation import PurgedTimeSeriesCV

        # Use reduced settings for testing with smaller gap
        # For 10k samples, we need much smaller splits
        cv = PurgedTimeSeriesCV(
            n_splits=TEST_CONFIG['n_folds'],
            gap_size=min(500, int(len(X) * 0.05)),  # Smaller gap for test data
            val_size=min(2000, int(len(X) * 0.2))   # Smaller validation set
        )

        validation_passed = True
        fold_info = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Check temporal ordering
            temporal_check = train_idx[-1] < val_idx[0]

            # Check gap
            gap_size = val_idx[0] - train_idx[-1] - 1
            gap_check = gap_size >= (len(X) * 0.04)  # At least 4% gap

            # Check no overlap
            overlap_check = len(np.intersect1d(train_idx, val_idx)) == 0

            fold_info.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'gap_size': gap_size,
                'temporal_ok': temporal_check,
                'gap_ok': gap_check,
                'overlap_ok': overlap_check
            })

            if not all([temporal_check, gap_check, overlap_check]):
                validation_passed = False

        self.validation_checks['cv_setup'] = {
            'passed': validation_passed,
            'folds': fold_info
        }

        # Display results
        for fold in fold_info:
            status = "✓" if all([fold['temporal_ok'], fold['gap_ok'], fold['overlap_ok']]) else "✗"
            print(f"  Fold {fold['fold']}: {status} Train={fold['train_size']}, "
                  f"Gap={fold['gap_size']}, Val={fold['val_size']}")

        return cv, validation_passed

    def test_feature_engineering(self, X, y):
        """Test feature engineering with observability."""
        print("\n🔧 Testing Feature Engineering...")
        start_time = time.time()

        from src.features.feature_engineer import FeatureEngineer

        # Use reduced settings
        fe = FeatureEngineer(
            n_top_interactions=TEST_CONFIG['n_interactions'],
            rolling_window=50,  # Smaller window for testing
            min_correlation=0.05
        )

        # Test on small subset
        X_subset = X.iloc[:1000]
        y_subset = y.iloc[:1000]

        # Fit and transform
        X_fe = fe.fit_transform(X_subset, y_subset)

        fe_time = time.time() - start_time
        self.timings['feature_engineering'] = fe_time

        # Validation checks
        feature_stats = {
            'original_features': X.shape[1],
            'engineered_features': X_fe.shape[1],
            'interactions_created': len(fe.selected_interactions),
            'time_taken': fe_time,
            'nan_count': X_fe.isna().sum().sum()
        }

        self.validation_checks['feature_engineering'] = feature_stats

        print(f"✓ Features: {feature_stats['original_features']} → "
              f"{feature_stats['engineered_features']} in {fe_time:.2f}s")
        print(f"  Top interactions: {fe.selected_interactions[:3]}")
        print(f"  NaN values: {feature_stats['nan_count']}")

        return fe

    def test_model_training(self, X, y, cv):
        """Test model training with minimal resources."""
        print("\n🚀 Testing Model Training (Minimal)...")

        from src.models.lgbm_model import LightGBMModel

        # Minimal LightGBM parameters
        test_params = {
            'objective': 'regression',
            'n_estimators': TEST_CONFIG['n_estimators'],
            'max_depth': 4,  # Shallower trees
            'learning_rate': 0.1,  # Higher learning rate for fewer trees
            'subsample': 0.8,
            'num_leaves': 15,
            'random_state': 42,
            'verbose': -1
        }

        model = LightGBMModel(params=test_params)

        # Train on first fold only for testing
        train_idx, val_idx = next(cv.split(X, y))
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Simple feature engineering for test
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )

        start_time = time.time()
        model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        train_time = time.time() - start_time

        # Get predictions
        val_pred = model.predict(X_val_scaled)

        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error

        metrics = {
            'r2_y1': r2_score(y_val['Y1'], val_pred[:, 0]),
            'r2_y2': r2_score(y_val['Y2'], val_pred[:, 1]),
            'rmse_y1': np.sqrt(mean_squared_error(y_val['Y1'], val_pred[:, 0])),
            'rmse_y2': np.sqrt(mean_squared_error(y_val['Y2'], val_pred[:, 1])),
            'train_time': train_time
        }

        metrics['weighted_r2'] = (metrics['r2_y1'] + metrics['r2_y2']) / 2

        # Check for overfitting
        train_val_gap_y1 = model.train_scores['Y1']['r2'] - model.val_scores['Y1']['r2']
        train_val_gap_y2 = model.train_scores['Y2']['r2'] - model.val_scores['Y2']['r2']
        metrics['train_val_gap'] = (train_val_gap_y1 + train_val_gap_y2) / 2

        self.results['model_test'] = metrics
        self.timings['model_training'] = train_time

        print(f"✓ Model trained in {train_time:.2f}s")
        print(f"  Y1 R²: {metrics['r2_y1']:.4f}")
        print(f"  Y2 R²: {metrics['r2_y2']:.4f}")
        print(f"  Weighted R²: {metrics['weighted_r2']:.4f}")
        print(f"  Train-Val Gap: {metrics['train_val_gap']:.4f}")

        # Warning checks
        if metrics['weighted_r2'] < 0.5:
            print("  ⚠️ Low R² - Check data quality or increase samples")
        if metrics['train_val_gap'] > 0.05:
            print("  ⚠️ Overfitting detected - Increase regularization")

        return model, val_pred, metrics

    def test_ensemble_methods(self):
        """Test ensemble functionality with mock predictions."""
        print("\n🎭 Testing Ensemble Methods...")

        from src.ensemble import MiddleAveragingEnsemble, compare_ensemble_methods

        # Create mock predictions for testing
        n_samples = 1000
        np.random.seed(42)

        # Simulate predictions from 3 models
        mock_predictions = {
            'lgbm': np.random.randn(n_samples, 2) * 0.5 + 1,
            'xgboost': np.random.randn(n_samples, 2) * 0.6 + 0.9,
            'neural_net': np.random.randn(n_samples, 2) * 0.7 + 0.95
        }

        # Create mock targets
        mock_targets = pd.DataFrame({
            'Y1': np.random.randn(n_samples) * 0.5 + 1,
            'Y2': np.random.randn(n_samples) * 0.5 + 1
        })

        # Test middle averaging
        middle_ens = MiddleAveragingEnsemble(keep_ratio=0.6)
        ensemble_pred = middle_ens.predict(mock_predictions)

        # Calculate diversity
        correlations = []
        models = list(mock_predictions.keys())
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                corr = np.corrcoef(
                    mock_predictions[models[i]].flatten(),
                    mock_predictions[models[j]].flatten()
                )[0, 1]
                correlations.append(corr)

        avg_correlation = np.mean(correlations)

        ensemble_stats = {
            'n_models': len(mock_predictions),
            'avg_correlation': avg_correlation,
            'ensemble_shape': ensemble_pred.shape,
            'diversity_ok': avg_correlation < 0.9
        }

        self.validation_checks['ensemble'] = ensemble_stats

        print(f"✓ Ensemble with {ensemble_stats['n_models']} models")
        print(f"  Average correlation: {ensemble_stats['avg_correlation']:.3f}")
        print(f"  Diversity check: {'✓' if ensemble_stats['diversity_ok'] else '✗'}")

        return ensemble_stats

    def generate_observability_report(self):
        """Generate comprehensive observability report."""
        print("\n" + "=" * 60)
        print("📊 OBSERVABILITY REPORT")
        print("=" * 60)

        # Timing summary
        print("\n⏱️ Performance Timings:")
        total_time = sum(self.timings.values())
        for component, duration in self.timings.items():
            percentage = (duration / total_time) * 100
            print(f"  {component}: {duration:.2f}s ({percentage:.1f}%)")
        print(f"  Total: {total_time:.2f}s")

        # Validation summary
        print("\n✅ Validation Checks:")
        all_passed = True
        for check_name, check_data in self.validation_checks.items():
            if isinstance(check_data, dict) and 'passed' in check_data:
                status = "✓" if check_data['passed'] else "✗"
                if not check_data['passed']:
                    all_passed = False
            else:
                status = "✓"
            print(f"  {check_name}: {status}")

        # Model performance
        if 'model_test' in self.results:
            print("\n📈 Model Performance:")
            metrics = self.results['model_test']
            print(f"  Weighted R²: {metrics['weighted_r2']:.4f}")
            print(f"  Train-Val Gap: {metrics['train_val_gap']:.4f}")

            # Quality gates
            print("\n🚦 Quality Gates:")
            gates = {
                'CV Score > 0.5 (test)': metrics['weighted_r2'] > 0.5,
                'Train-Val Gap < 0.1': metrics['train_val_gap'] < 0.1,
                'All validations passed': all_passed
            }

            for gate, passed in gates.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {gate}: {status}")

        # Save report
        report = {
            'timings': self.timings,
            'validation_checks': self.validation_checks,
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        report_path = Path('results') / 'test_report.json'
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Report saved to: {report_path}")

        return all_passed

    def create_diagnostic_plots(self, y_true, y_pred):
        """Create diagnostic plots for observability."""
        print("\n📉 Generating Diagnostic Plots...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Predictions vs Actual for Y1
        axes[0, 0].scatter(y_true['Y1'], y_pred[:, 0], alpha=0.5, s=10)
        axes[0, 0].plot([y_true['Y1'].min(), y_true['Y1'].max()],
                        [y_true['Y1'].min(), y_true['Y1'].max()],
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Y1')
        axes[0, 0].set_ylabel('Predicted Y1')
        axes[0, 0].set_title('Y1: Predictions vs Actual')

        # Plot 2: Predictions vs Actual for Y2
        axes[0, 1].scatter(y_true['Y2'], y_pred[:, 1], alpha=0.5, s=10)
        axes[0, 1].plot([y_true['Y2'].min(), y_true['Y2'].max()],
                        [y_true['Y2'].min(), y_true['Y2'].max()],
                        'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Y2')
        axes[0, 1].set_ylabel('Predicted Y2')
        axes[0, 1].set_title('Y2: Predictions vs Actual')

        # Plot 3: Residuals for Y1
        residuals_y1 = y_true['Y1'] - y_pred[:, 0]
        axes[1, 0].hist(residuals_y1, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Y1 Residuals (mean={residuals_y1.mean():.4f})')

        # Plot 4: Residuals for Y2
        residuals_y2 = y_true['Y2'] - y_pred[:, 1]
        axes[1, 1].hist(residuals_y2, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Y2 Residuals (mean={residuals_y2.mean():.4f})')

        plt.tight_layout()

        # Save plot
        plot_path = Path('results') / 'diagnostic_plots.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Plots saved to: {plot_path}")

        # Don't show in test mode to save resources
        plt.close()

    def run_minimal_test(self):
        """Run complete minimal test pipeline."""
        print("\n🔬 RUNNING MINIMAL TEST PIPELINE")
        print("=" * 60)

        try:
            # 1. Load test data
            X, y = self.load_test_data()

            # 2. Validate CV setup
            cv, cv_valid = self.validate_cv_setup(X, y)
            if not cv_valid:
                print("⚠️ CV validation failed - check implementation!")

            # 3. Test feature engineering
            fe = self.test_feature_engineering(X, y)

            # 4. Test model training
            model, predictions, metrics = self.test_model_training(X, y, cv)

            # 5. Test ensemble methods
            ensemble_stats = self.test_ensemble_methods()

            # 6. Generate diagnostic plots
            # Use subset for plotting
            plot_idx = np.random.choice(len(predictions), min(500, len(predictions)), replace=False)
            y_subset = y.iloc[plot_idx]
            pred_subset = predictions[plot_idx]
            self.create_diagnostic_plots(y_subset, pred_subset)

            # 7. Generate final report
            all_passed = self.generate_observability_report()

            print("\n" + "=" * 60)
            if all_passed:
                print("✅ ALL TESTS PASSED - Pipeline is working correctly!")
                print("You can proceed with full training on your PC.")
            else:
                print("⚠️ Some tests failed - Review the report above.")
                print("Fix issues before running full pipeline.")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        return all_passed


def main():
    """Main entry point for testing."""
    # Check if data exists
    if not Path('data/train.csv').exists():
        print("❌ Error: data/train.csv not found!")
        print("Please ensure the data file is in the correct location.")
        return

    # Run minimal test
    tester = MinimalTestPipeline(test_mode=True)
    success = tester.run_minimal_test()

    if success:
        print("\n📋 Next Steps:")
        print("1. Review results/test_report.json for detailed metrics")
        print("2. Check results/diagnostic_plots.png for visual validation")
        print("3. If satisfied, run full pipeline on your PC:")
        print("   python src/train_ensemble.py")

    return success


if __name__ == "__main__":
    main()