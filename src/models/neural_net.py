"""
Neural network models for return prediction
"""

import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from .baseline import BaseModel

logger = logging.getLogger(__name__)

class MLPNet(nn.Module):
    """
    Multi-Layer Perceptron architecture
    Simple feedforward neural network with configurable layers
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        """
        Args
            input_dim: number of input features
            hidden_dims: list of hidden layer sizes (for instance [64,32])
            dropout: dropout rate for regularization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer (single neuron for binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralNetModel(BaseModel):
    """
    Neural network classifier using PyTorch
    Wraps MLPNet with the standard interface
    """

    def __init__(
            self,
            hidden_dims: list[int] = [32,16],
            dropout: float = 0.2,
            learning_rate: float = 0.001,
            batch_size: int = 64,
            epochs: int = 100,
            early_stopping_patience: int = 10,
            scale_features: bool = True,
            verbose: bool = False
    ):
        """
        Args:
            hidden_dims: Sizes of hidden layers
            dropout: Dropout rate
            learning_rate: Adam optimizer learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Stop if no improvement for this many epochs
            scale_features: Whether to standardize inputs
            verbose: Print training progress
        """
        super().__init__(scale_features)

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.net = None
        self.training_history = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # scale features
        if self.scale_features:
            X = self.scaler.fit_transform(X)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize network
        input_dim = X_tensor.shape[1]
        self.net = MLPNet(input_dim, hidden_dims=self.hidden_dims, dropout=self.dropout)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Training loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output = self.net(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.training_history.append(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        self.is_fitted = True
        logger.info(f"NeuralNetModel fitted on {len(y)} samples, "
                   f"stopped at epoch {len(self.training_history)}")


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("NeuralNetModel not fitted. Call .fit() first")

        if self.scale_features:
            X = self.scaler.transform(X)

        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            proba = self.net(X_tensor).numpy().flatten()

        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def plot_training_history(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.plot(self.training_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.show()
