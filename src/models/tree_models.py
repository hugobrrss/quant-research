"""
Tree-base models for return prediction
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Optional

from .baseline import BaseModel

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """ Random forest classifier """

    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: Optional[int] = 5, # here Optional is either 5 | None, so the feature can be "disabled"
            min_samples_leaf: int = 50,
            scale_features: bool = False
):
        """
        Args:
            - n_estimators: number of tress
            - max_depth: max depth of trees (None=unlimited, likely overfit)
            - min_samples_split: min samples required at leaf node
        """
        super().__init__(scale_features)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1 # use all CPU cores
        )

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError('Model has not been fitted yet')

        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)

        return importance

class GradientBoostingModel(BaseModel):
    """ Gradient boosting classifier """
    def __init__(
            self,
            n_estimators: int = 100,
            max_depth: int = 3,
            learning_rate: float = 0.1,
            min_samples_leaf: int = 50,
            scale_features: bool = False
    ):
        """
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (keep small for boosting)
            learning_rate: Shrinkage parameter (smaller = more trees needed, but more robust)
            min_samples_leaf: Minimum samples at leaf
        """
        super().__init__(scale_features)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError('Model has not been fitted yet')

        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)

        return importance


