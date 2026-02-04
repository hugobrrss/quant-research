"""
Baseline models for return prediction
"""

import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple

logger = logging.getLogger(__name__)

class BaseModel:
    """
    Base class for prediction models
    This handles:
        - feature scaling
        - fitting and prediction
        - evaluation metrics
    """

    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.scaler = StandardScaler() if self.scale_features else None
        self.model = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self.scale_features:
            X_train = self.scaler.fit_transform(X_train)

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(f"{self.__class__.__name__} fitted on {len(y_train)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise RuntimeError(f"Model not fitted, call .fit() first")

        if self.scale_features:
            X = self.scaler.transform(X)    # here transform() and not fit_transform() to prevent data leakage (uses estimated mean and std from training data only)

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise RuntimeError(f"Model not fitted, call .fit() first")

        if self.scale_features:
            X = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:,1]
        else:
            return self.model.decision_function(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Returns dict with several performance metrics"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'n_samples': len(y),
            'pct_predicted_up': y_pred.mean(),
            'pct_actual_up': y.mean()
        }

        return metrics

    def evaluate_and_log(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = 'Test') -> dict:
        """Evaluate and print results"""
        metrics = self.evaluate(X, y)

        logger.info(f"\n{dataset_name} Results for {self.__class__.__name__}:")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC:   {metrics['roc_auc']:.4f}")
        logger.info(f"Predicted {metrics['pct_predicted_up']:.1%} up, Actual {metrics['pct_actual_up']:.1%} up")

        return metrics

class LogisticRegressionModel(BaseModel):
    """
    Logistic regression baseline model
    Supports:
        - Standard unregularized logistic regression
        - Ridge
        - Lasso
        - ElasticNet
    """

    def __init__(self, scale_features: bool = True, C: float = 1.0, l1_ratio: float = 0):
        """
        Args:
            - scale_features: whether to standardise features
            - C: Inverse regularisation strength (set C=np.inf for unregularized logistic regression)
            - l1_ratio: L1 regularisation strength (0 for pure L2 penalty, 1 for pure L1 penalty, and anything between 0 and 1 for Elastic-Net)
        """
        super().__init__(scale_features)
        self.l1_ratio = l1_ratio

        # saga solver is required for l1_ratio > 0
        if self.l1_ratio > 0:
            solver = 'saga'
        else:
            solver = 'lbfgs'
        self.model = LogisticRegression(l1_ratio=l1_ratio, solver=solver, C=C, max_iter=1000, random_state=42)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(f"Model not fitted, call .fit() first")

        if not self.scale_features:
            logger.warning(f"Model fitted on features without scaling, features importance based on coefficients not valid")

        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_[0]
        })
        importance['abs_coefficient'] = importance['coefficient'].abs()
        importance = importance.sort_values('abs_coefficient', ascending=False)

        return importance

def compare_models(
        models: list[BaseModel],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> pd.DataFrame:
    """
    Train and compare multiple models
    Returns metrics for each model
    """
    results = []
    for model in models:
        model.fit(X_train, y_train)

        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)

        results.append({
            'model': model.__class__.__name__,
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy'],
            'train_auc': train_metrics['roc_auc'],
            'test_auc': test_metrics['roc_auc'],
            'overfit_gap': train_metrics['accuracy'] - test_metrics['accuracy']
        })

    return pd.DataFrame(results)