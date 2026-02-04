"""
Dataset preparation for prediction models
"""
import logging

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class PredictionDataset:
    """"
    Prepares data for return prediction models.
    Handles:
        - target variables creation
        - features/target alignment
        - train/test split
    """

    def __init__(
            self,
            df: pd.DataFrame,
            features_col: list[str],
            target_horizon: int=1,
            target_type: str = 'binary'
    ):
        """
        Args:
            df: DatFrame with fetures and price data
            features_col: List of features column names
            target_horizon: Target horizon for prediction (default is one day)
            target_type: 'binary' or 'continuous'
        """
        self.df = df.copy()
        self.features_col = features_col
        self.target_horizon = target_horizon
        self.target_type = target_type
        self._create_target()
        self._clean_data()

    def _create_target(self):
        self.df['forward_return'] = self.df['Close'].pct_change(self.target_horizon).shift(-self.target_horizon)

        if self.target_type == 'continuous':
            self.df['target'] = self.df['forward_return']
        elif self.target_type == 'binary':
            self.df['target'] = (self.df['forward_return'] > 0).astype(int)
        else:
            raise ValueError(f"Target type {self.target_type} not recognized")

        logger.info(f"Created target type {self.target_type} with target horizon {self.target_horizon} day")

    def _clean_data(self):
        cols_needed = self.features_col + ['target']
        before = len(self.df)
        self.df = self.df.dropna(subset=cols_needed)
        after = len(self.df)
        logger.info(f"Removed {before - after} rows from dataframe ({after} rows remaining)")

    def get_train_test_split(
            self,
            test_size: float=0.2,
            method: str = 'time'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        Args:
            test_size: Fraction of data to be used for testing
            method: 'time' (last x%) or 'random' (shuffled)
        Returns:
            X_train, X_test, y_train, y_test
        """
        if method == 'random':
            logger.warning("Random split used - introduce lookahead bias in time series")
            from sklearn.model_selection import train_test_split
            X = self.df[self.features_col]
            y = self.df['target']
            return train_test_split(X, y, test_size=test_size, random_state=42)

        elif method == 'time':
            split_idx = int(len(self.df) * (1-test_size))
            train = self.df.iloc[:split_idx]
            test = self.df.iloc[split_idx:]

            X_train = train[self.features_col]
            X_test = test[self.features_col]
            y_train = train['target']
            y_test = test['target']

            return X_train, X_test, y_train, y_test

        else:
            raise ValueError(f"Method {method} not recognized")

    def get_walk_forward_splits(
            self,
            n_splits: int = 5,
            train_size: float = 0.6,
            test_size: float = 0.1
    ) -> list[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:

        """Generates walk-forwards validation splits"""

        splits = []
        n = len(self.df)

        for i in range(n_splits):
            train_start = int(i * n * (1 - train_size - test_size) / n_splits)
            train_end = train_start + int(n * train_size)
            test_end = train_end + int(n * test_size)

            if test_end > n:
                break

            train = self.df.iloc[train_start:train_end]
            test = self.df.iloc[train_end:test_end]

            X_train = train[self.features_col]
            X_test = test[self.features_col]
            y_train = train['target']
            y_test = test['target']

            splits.append((X_train, X_test, y_train, y_test))

            logger.info(f"Split {i+1}: train {train.index.min().date()} to {train.index.max().date()}, "
                        f"test {test.index.min().date()} to {test.index.max().date()}")

        return splits


    def get_class_balance(self) -> dict:
        """Check class balance for binary classification"""
        counts = self.df['target'].value_counts()
        total = len(self.df)

        balance = {
            'down (0)': counts.get(0,0),
            'up (1)': counts.get(1,0),
            'up_ratio': counts.get(1,0) / total
        }

        logger.info(f"Class balance: {balance['up (1)']} up ({balance['up_ratio']:.1%}), "
                   f"{balance['down (0)']} down ({1-balance['up_ratio']:.1%})")
        return balance
