"""
Neural Network Model with High Regularization

Multi-layer perceptron with 0.4 dropout for preventing overfitting.
Implements correlation-aware loss for multi-target learning.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class MultiTargetMLP(nn.Module):
    """
    Multi-target neural network with high dropout regularization.

    Critical: 0.4 dropout proven essential from Jane Street analysis.
    """

    def __init__(self, input_dim: int, hidden_dims: list = [32, 64, 32], dropout: float = 0.4):
        """
        Initialize the neural network.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate (0.4 recommended for financial data)
        """
        super(MultiTargetMLP, self).__init__()

        self.dropout_rate = dropout
        layers = []

        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer for 2 targets (Y1, Y2)
        layers.append(nn.Linear(prev_dim, 2))

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


class NeuralNetModel:
    """
    Wrapper class for PyTorch neural network with scikit-learn-like interface.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Neural Network model.

        Args:
            params: Model parameters dictionary
        """
        self.default_params = {
            'layers': [32, 64, 32],
            'dropout': 0.4,  # High dropout critical for preventing overfitting
            'activation': 'relu',
            'batch_size': 512,
            'epochs': 100,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # Merge provided params with defaults to avoid missing keys
        self.params = self.default_params.copy()
        if params is not None:
            self.params.update(params)
        self.model = None
        self.input_dim = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.train_scores = {}
        self.val_scores = {}

    def correlation_aware_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function that leverages Y1-Y2 correlation.

        Args:
            predictions: Model predictions (batch_size, 2)
            targets: True values (batch_size, 2)

        Returns:
            Combined loss value
        """
        # Individual MSE losses
        y1_loss = F.mse_loss(predictions[:, 0], targets[:, 0])
        y2_loss = F.mse_loss(predictions[:, 1], targets[:, 1])

        # Correlation-aware loss (average of Y1 and Y2)
        avg_pred = (predictions[:, 0] + predictions[:, 1]) / 2
        avg_true = (targets[:, 0] + targets[:, 1]) / 2
        corr_loss = F.mse_loss(avg_pred, avg_true)

        # Weighted combination
        total_loss = 0.4 * y1_loss + 0.4 * y2_loss + 0.2 * corr_loss

        return total_loss

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.DataFrame] = None):
        """
        Train the neural network model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Self for chaining
        """
        print("\nTraining Neural Network...")

        # Convert to numpy arrays
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train[['Y1', 'Y2']].values.astype(np.float32)

        if X_val is not None and y_val is not None:
            X_val_np = X_val.values.astype(np.float32)
            y_val_np = y_val[['Y1', 'Y2']].values.astype(np.float32)
        else:
            X_val_np, y_val_np = None, None

        # Store input dimension
        self.input_dim = X_train_np.shape[1]

        # Create model
        self.model = MultiTargetMLP(
            input_dim=self.input_dim,
            hidden_dims=self.params['layers'],
            dropout=self.params['dropout']
        ).to(self.params['device'])

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_np),
            torch.FloatTensor(y_train_np)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params['batch_size'],
            shuffle=True
        )

        if X_val_np is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_np),
                torch.FloatTensor(y_val_np)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.params['batch_size'],
                shuffle=False
            )

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.params['epochs']
        )

        # Training loop
        patience_counter = 0
        best_epoch = 0

        for epoch in range(self.params['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.params['device'])
                batch_y = batch_y.to(self.params['device'])

                optimizer.zero_grad()
                predictions = self.model(batch_x)
                loss = self.correlation_aware_loss(predictions, batch_y)

                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)

            # Validation phase
            if X_val_np is not None:
                self.model.eval()
                val_loss = 0
                val_batches = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.params['device'])
                        batch_y = batch_y.to(self.params['device'])

                        predictions = self.model(batch_x)
                        loss = self.correlation_aware_loss(predictions, batch_y)

                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches
                self.val_losses.append(avg_val_loss)

                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_model_state = self.model.state_dict()
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1

                if patience_counter >= self.params['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.params['epochs']}: "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, "
                          f"LR: {scheduler.get_last_lr()[0]:.6f}")

            scheduler.step()

        # Load best model state
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {best_epoch + 1}")

        # Calculate final scores
        self._calculate_scores(X_train_np, y_train_np, X_val_np, y_val_np)

        return self

    def _calculate_scores(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Calculate R² scores for training and validation sets."""
        # Training scores
        train_pred = self.predict_numpy(X_train)
        self.train_scores = {
            'Y1': {
                'r2': r2_score(y_train[:, 0], train_pred[:, 0]),
                'rmse': np.sqrt(mean_squared_error(y_train[:, 0], train_pred[:, 0]))
            },
            'Y2': {
                'r2': r2_score(y_train[:, 1], train_pred[:, 1]),
                'rmse': np.sqrt(mean_squared_error(y_train[:, 1], train_pred[:, 1]))
            }
        }

        # Validation scores
        if X_val is not None and y_val is not None:
            val_pred = self.predict_numpy(X_val)
            self.val_scores = {
                'Y1': {
                    'r2': r2_score(y_val[:, 0], val_pred[:, 0]),
                    'rmse': np.sqrt(mean_squared_error(y_val[:, 0], val_pred[:, 0]))
                },
                'Y2': {
                    'r2': r2_score(y_val[:, 1], val_pred[:, 1]),
                    'rmse': np.sqrt(mean_squared_error(y_val[:, 1], val_pred[:, 1]))
                }
            }

            # Check for overfitting
            for target in ['Y1', 'Y2']:
                train_val_gap = self.train_scores[target]['r2'] - self.val_scores[target]['r2']
                if train_val_gap > 0.05:
                    print(f"⚠️ Warning: Large train-val gap ({train_val_gap:.4f}) for {target}")

            print(f"\nFinal Scores:")
            print(f"Y1 - Train R²: {self.train_scores['Y1']['r2']:.4f}, "
                  f"Val R²: {self.val_scores['Y1']['r2']:.4f}")
            print(f"Y2 - Train R²: {self.train_scores['Y2']['r2']:.4f}, "
                  f"Val R²: {self.val_scores['Y2']['r2']:.4f}")

    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions from numpy array.

        Args:
            X: Features array

        Returns:
            Predictions array
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.params['device'])
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for both Y1 and Y2.

        Args:
            X: Features DataFrame

        Returns:
            Array of shape (n_samples, 2) with Y1 and Y2 predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")

        X_np = X.values.astype(np.float32)
        return self.predict_numpy(X_np)

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
        weighted_r2 = (r2_y1 + r2_y2) / 2
        return weighted_r2

    def save_model(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first!")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
            'input_dim': self.input_dim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.params['device'])

        self.params = checkpoint['params']
        self.input_dim = checkpoint['input_dim']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        self.model = MultiTargetMLP(
            input_dim=self.input_dim,
            hidden_dims=self.params['layers'],
            dropout=self.params['dropout']
        ).to(self.params['device'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filepath}")