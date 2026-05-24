from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .feature_prep import build_features


class RandomForestModel:
    """Random Forest classifier for the recipe rating prediction.

    Uses the same preprocessing pipeline as the baseline.
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
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        class_names: dict[int, str] | None = None,
    ) -> None:
        """Initialise the Random Forest model.

        Args:
            numeric_block_columns (list[str]): Column names of the numeric block
                of the sparse matrix, in order. Passed straight to the preprocessor.
            n_estimators (int, optional): Number of trees. Defaults to 200.
            max_depth (int | None, optional): Maximum tree depth. None grows
                trees until leaves are pure. Defaults to None.
            random_state (int, optional): Random state for reproducibility.
                Defaults to 42.
            class_names (dict[int, str] | None, optional): Mapping from integer
                class label to name. Falls back to DEFAULT_CLASS_NAMES
                when None.
        """
        self._numeric_block_columns = numeric_block_columns
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self._class_names = class_names or self.DEFAULT_CLASS_NAMES

        self._pipe: Pipeline | None = None

    def fit(self, X: sparse.csr_matrix, y: pd.Series) -> RandomForestModel:
        """Fit the Random Forest pipeline on training data.

        Args:
            X (sparse.csr_matrix): Feature matrix produced by BuildFeatureMatrix.
            y (pd.Series): Integer class labels.

        Returns:
            RandomForestModel: Fitted estimator (self).
        """
        self._pipe = Pipeline(
            [
                (
                    "preprocessing",
                    build_features(self._numeric_block_columns),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=self._n_estimators,
                        max_depth=self._max_depth,
                        class_weight="balanced",
                        random_state=self._random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self._pipe.fit(X, y)
        return self

    def predict(
        self,
        X: sparse.csr_matrix,
        output_class_names: bool = False,
        return_proba: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict rating class for each sample.

        Args:
            X (sparse.csr_matrix): Feature matrix produced by BuildFeatureMatrix.
            output_class_names (bool, optional): If True, return human-readable
                class names instead of integer labels. Defaults to False.
            return_proba (bool, optional): If True, also return the class
                probability matrix (n_samples x n_classes). Defaults to False.

        Raises:
            RuntimeError: If called before fit.

        Returns:
            np.ndarray: Predicted labels (n_samples,).
            If return_proba is True, returns (labels, probs) where probs has
            shape (n_samples, n_classes).
        """
        if self._pipe is None:
            raise RuntimeError("Call fit before predict.")

        labels = self._pipe.predict(X)

        if output_class_names:
            labels = np.array([self._class_names[int(c)] for c in labels])
        else:
            labels = labels.astype(int)

        if return_proba:
            probs = self._pipe.predict_proba(X)
            return labels, probs

        return labels