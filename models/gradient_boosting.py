from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.pipeline import Pipeline

from .feature_prep import build_features

class GradientBoostingModel:
    """LightGBM gradient boosting classifier for the recipe rating prediction.

    The same preprocessing pipeline as the baseline and Random forest is used. 
    Class imbalance is handled via class_weight='balanced' instead of
    manual undersampling.

    Attributes:
        DEFAULT_CLASS_NAMES: Mapping from integer class label to name.
    """

    DEFAULT_CLASS_NAMES: dict[int, str] = {0: "<= 4.0", 1: "4.5", 2: "5.0"}

    def __init__(
        self,
        numeric_block_columns: list[str],
        *,
        n_estimators: int = 300,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        random_state: int = 42,
        max_depth: int = -1,
        min_child_samples: int = 20,
        class_names: dict[int, str] | None = None,
    ) -> None:
        """Initialise the Gradient Boosting model.

        Args:
            numeric_block_columns (list[str]): Column names of the numeric block
                of the sparse matrix. The column names are in order. This is passed
                straight to the preprocessor.
            n_estimators (int): Number of boosting rounds. Defaults to 300.
            learning_rate (float): Shrinkage rate. A smaller value requires
                more trees but it usually generalizes better. Defaults to 0.05.
            num_leaves (int): Maximum leaves per tree. Controls model
                complexity together with max_depth. Defaults to 31.
            max_depth (int): Maximum tree depth. -1 means that there is no limit.
                The is controlled by num_leaves. Defaults to -1.
            min_child_samples (int): Minimum samples in a leaf. When it has 
                higher values, this will reduce overfitting. Defaults to 20.
            random_state (int): Random state for reproducibility. Defaults to 42.
            class_names (dict[int, str] | None, optional): Mapping from integer
                class label to name. Falls back to DEFAULT_CLASS_NAMES when None.
        """
        self._numeric_block_columns = numeric_block_columns
        self._n_estimators = n_estimators
        self._num_leaves = num_leaves
        self._learning_rate = learning_rate
        self._random_state = random_state
        self._max_depth = max_depth
        self._min_child_samples = min_child_samples
        self._class_names = class_names or self.DEFAULT_CLASS_NAMES

        self._pipe: Pipeline | None = None

    def fit(self, X: sparse.csr_matrix, y: pd.Series) -> GradientBoostingModel:
        """This function will fit the Gradient Boosting pipeline on training data.

        Args:
            X (sparse.csr_matrix): This is a feature matrix produced by BuildFeatureMatrix.
            y (pd.Series): Integer class labels.

        Returns:
            GradientBoostingModel: Fitted estimator.
        """
        self._pipe = Pipeline(
            [
                (
                    "preprocessing",
                    build_features(self._numeric_block_columns),
                ),
                (
                    "gbm",
                    LGBMClassifier(
                        n_estimators=self._n_estimators,
                        num_leaves=self._num_leaves,
                        learning_rate=self._learning_rate,
                        random_state=self._random_state,
                        max_depth=self._max_depth,
                        min_child_samples=self._min_child_samples,
                        class_weight="balanced",
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )
        self._pipe.fit(X, y)
        return self

    def predict(self, X: sparse.csr_matrix, output_class_names: bool = False, 
                return_probability: bool = False,) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """This function predicts the rating class for each sample.

        Args:
            X (sparse.csr_matrix): Feature matrix produced by BuildFeatureMatrix.
            output_class_names (bool): If it is True, the human-readable
                class names are returned instead of the integer labels. Defaults to False.
            return_probability (bool): If True, this also returns the class
                probability matrix (n_samples x n_classes). Defaults to False.

        Raises:
            RuntimeError: If called before fit.

        Returns:
            np.ndarray: Predicted labels (n_samples,).
            If return_probability is True, returns (labels, probs) where probs has the 
            shape of (n_samples, n_classes).
        """
        if self._pipe is None:
            raise RuntimeError("Call fit before predict.")

        labels = self._pipe.predict(X)

        if output_class_names:
            labels = np.array([self._class_names[int(c)] for c in labels])
        else:
            labels = labels.astype(int)

        if return_probability:
            probability = self._pipe.predict_proba(X)
            return labels, probability

        return labels
    
    @property
    def classes_(self):
        return self._class_names