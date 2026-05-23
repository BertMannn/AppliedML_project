from __future__ import annotations

import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy import sparse

from imblearn.under_sampling import RandomUnderSampler
from itertools import combinations


NUMERIC_FEATURE_NAMES = (
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
    "CookTimeMinutes",
    "PrepTimeMinutes",
    "TotalTimeMinutes",
    "KeywordCount",
    "IngredientCount",
    "InstructionStepCount",
    "DescriptionLength",
    "RecipeServings",
)


def build_logres_features(numeric_block_columns: list[str]) -> ColumnTransformer:
    """Build preprocessing ColumnTransformer for sparse matrix using in our logres baseline

    Args:
        numeric_block_columns (list[str]): column names of the numeric block of the sparse matrix, in order

    Returns:
        ColumnTransformer: Can now use sparse input
    """
    numeric_names = set(NUMERIC_FEATURE_NAMES)
    numeric_indices = [
        i for i, c in enumerate(numeric_block_columns) if c in numeric_names
    ]
    categorical_indices = [
        i
        for i, c in enumerate(numeric_block_columns)
        if c.startswith("RecipeCategory_")
    ]

    return ColumnTransformer(
        [
            # with_mean = False, because we are using sparse matrices
            ("numeric", StandardScaler(with_mean=False), numeric_indices),
            ("categorical", "passthrough", categorical_indices),
        ],
        remainder="passthrough",
    )


class BaselineLR:
    """Classes enabling us to get the baseline estimations of model performance

    We designed our own model, called via "fit" and "predict", then we also have
    standard sklearn model. The standard sklearn models can be fitted using "fit_models"
    and predictions can be called using "predict_all". 
    """
    def __init__(
        self,
        numeric_block_columns: list[str],
        *,
        max_iter: int = 1000,
        random_state: int = 42,
        rating_classes: tuple[int, ...] = (0, 1, 2),
        minority_class: int = 0,
        class_names: dict = {0: "<= 4.0", 1: "4.5", 2: "5.0"},
    ):
        """Init the class with the base values needed

        Args:
            numeric_block_columns (list[str]): column names of the numeric block of the sparse
            matrix, in order.
            random_state (int, optional): random state, used for reproduciblity. Defaults to 42.
            rating_classes (tuple[int, ...], optional): we have three classes, encoded to
            0 = <= 4.0, 1 = 4.5, 2 = 5.0 rating. Defaults to (0,1,2).
            minority_class (int, optional): <= 4.o is a heavy minority class,
            so we need to undersample it. Defaults to 0.
            class_names (dict): Dict mapping class encoding to class names, only used
            for prediction to show the name
        """
        self._numeric_block_columns = numeric_block_columns
        self._rating_classes = rating_classes
        self._random_state = random_state
        self._minority_class = minority_class
        self._class_names = class_names
        self._max_iter = max_iter

    def _undersample(
        self, X: sparse.csr_matrix, y: pd.Series
    ) -> tuple[pd.DataFrame, ...]:
        """Undersample majority class, this is needed for the <= 4.0 class.

        Args:
            X (sparse.csr_matrix): observations
            y (pd.Series): labels

        Returns:
            tuple[pd.DataFrame]: Undersampled X and y
        """
        return RandomUnderSampler(
            sampling_strategy="auto", random_state=self._random_state
        ).fit_resample(X, y)

    def fit(self, X: sparse.csr_matrix, y: pd.Series) -> BaselineLR:
        """Fit all three one-vs-one models

        Args:
            X (sparse.csr_matrix): observations
            y (pd.Series): labels

        Returns:
            BaselineOvO: Fitted estimator
        """
        self.models_ = {}
        for c1, c2 in combinations(self._rating_classes, 2):
            m = y.isin([c1, c2]).to_numpy()
            Xp, yp = X[m], y[m]

            if self._minority_class in (c1, c2):
                Xp, yp = self._undersample(Xp, yp)

            pipe = Pipeline(
                [
                    (
                        "preprocessing",
                        build_logres_features(self._numeric_block_columns),
                    ),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=self._max_iter, class_weight="balanced"
                        ),
                    ),
                ]
            )

            self.models_[(c1, c2)] = pipe.fit(Xp, yp)

        return self

    def predict(
        self,
        X: sparse.csr_matrix,
        output_class_names: bool = False,
        return_proba: bool = False,
    ) -> np.ndarray:
        """Predict class based on observation.

        Args:
            X (sparse.csr_matrix): observations
            output_class_names (bool, optional): if True, return human-readable class names;
                otherwise return integer class labels.. Defaults to False.
            return_proba (bool, optional): if True, also return the full probability
                distribution over classes (shape: n_samples x n_classes).. Defaults to False.

        Returns:
            np.ndarray: predicted labels (shape: n_samples,).
            If return_proba=True, returns a tuple (labels, probs) where probs is
            the softmax-normalized probability over all classes.
        """
        n = X.shape[0]
        classes = list(self._rating_classes)
        class_idx = {c: i for i, c in enumerate(classes)}

        votes = np.zeros((n, len(classes)))

        # Use probability as a tiebreaker
        proba_sum = np.zeros((n, len(classes)))

        for (c1, c2), pipe in self.models_.items():
            pred = pipe.predict(X)
            for c in (c1, c2):
                votes[pred == c, class_idx[c]] += 1

            proba = pipe.predict_proba(X)
            for j, c in enumerate(pipe.named_steps["lr"].classes_):
                proba_sum[:, class_idx[c]] += proba[:, j]

        # Rank first on # votes, then on probability
        ranked = np.lexsort((proba_sum, votes), axis=1)

        # lexsort goes from small to large, grab last element
        winners = ranked[:, -1]
        labels = np.array([classes[i] for i in winners])

        if output_class_names:
            labels = np.array([self._class_names[int(c)] for c in labels])
        else:
            labels = labels.astype(int)

        if return_proba:
            # Softmax over aggregated pairwise probabilities
            exp = np.exp(proba_sum)
            probs = exp / exp.sum(axis=1, keepdims=True)
            return labels, probs

        return labels

    def fit_models(self, X: sparse.csr_matrix, y: pd.Series) -> BaselineLR:
        """Fit standard models given by sklearn, to compare to our own model

        Args:
            X (sparse.csr_matrix): observations
            y (pd.Series): labels

        Returns:
            BaselineLR: fitted models, stored in self._pipes
        """

        base_lr = LogisticRegression(max_iter=self._max_iter, class_weight="balanced")

        self._pipes = {
            "standard": Pipeline(
                [
                    (
                        "preprocessing",
                        build_logres_features(self._numeric_block_columns),
                    ),
                    ("lr", base_lr),
                ]
            ),
            "ovr": Pipeline(
                [
                    (
                        "preprocessing",
                        build_logres_features(self._numeric_block_columns),
                    ),
                    ("lr", OneVsRestClassifier(base_lr)),
                ]
            ),
            "ovo": Pipeline(
                [
                    (
                        "preprocessing",
                        build_logres_features(self._numeric_block_columns),
                    ),
                    ("lr", OneVsOneClassifier(base_lr)),
                ]
            ),
        }

        for _, pipe in self._pipes.items():
            pipe.fit(X, y)

        return self

    def predict_all(self, X: sparse.csr_matrix) -> tuple[np.ndarray, ...]:
        """Predict the labels using the standard models

        Args:
            X (sparse.csr_matrix): observations

        Returns:
            tuple[np.ndarray, ...]: The predicted labels by the three models
        """
        results = {}
        classes = list(self._rating_classes)
        class_idx = {c: i for i, c in enumerate(classes)}
        n = X.shape[0]
        for strategy_name, pipe in self._pipes.items():
            votes = np.zeros((n, len(classes)))
            scores_sum = np.zeros((n, len(classes)))

            pred = pipe.predict(X)
            for c in classes:
                votes[pred == c, class_idx[c]] += 1

            # Tiebreaker: Use probabilities if available, otherwise decision function
            lr_step = pipe.named_steps["lr"]
            if hasattr(lr_step, "predict_proba"):
                scores = pipe.predict_proba(X)
            else:
                # OneVsOne uses decision_function for confidence scores
                scores = pipe.decision_function(X)

            for j, c in enumerate(lr_step.classes_):
                scores_sum[:, class_idx[c]] += scores[:, j]

            ranked = np.lexsort((scores_sum, votes), axis=1)
            winners = ranked[:, -1]
            results[strategy_name] = np.array([classes[i] for i in winners])

        return results["standard"], results["ovr"], results["ovo"]
