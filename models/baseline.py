from __future__ import annotations

import pandas as pd
import numpy as np


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from imblearn.under_sampling import RandomUnderSampler
from itertools import combinations


def build_logres_features(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
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
    ]
    cat_features = [c for c in features.columns if c.startswith("RecipeCategory_")]

    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", "passthrough", cat_features),
        ]
    )


class BaselineOvO:
    def __init__(
        self,
        *,
        random_state: int = 42,
        rating_classes: tuple[int, ...] = (0, 1, 2),
        minority_class: int = 0,
        class_names: dict = {0: "<= 4.0", 1: "4.5", 2: "5.0"},
    ):
        """Init the class with the base values needed

        Args:
            random_state (int, optional): random state, used for reproduciblity. Defaults to 42.
            rating_classes (tuple[int, ...], optional): we have three classes, encoded to
            0 = <= 4.0, 1 = 4.5, 2 = 5.0 rating. Defaults to (0,1,2).
            minority_class (int, optional): <= 4.o is a heavy minority class,
            so we need to undersample it. Defaults to 0.
            class_names (dict): Dict mapping class encoding to class names, only used
            for prediction to show the name
        """
        self._rating_classes = rating_classes
        self._random_state = random_state
        self._minority_class = minority_class
        self._class_names = class_names

    def _undersample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, ...]:
        """Undersample majority class, this is needed for the <= 4.0 class.

        Args:
            X (pd.DataFrame): observations
            y (pd.DataFrame): labels

        Returns:
            tuple[pd.DataFrame]: Undersampled X and y
        """
        return RandomUnderSampler(
            sampling_strategy="auto", random_state=self._random_state
        ).fit_resample(X, y)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaselineOvO:
        """Fit all three one-vs-one models

        Args:
            X (pd.DataFrame): observations
            y (pd.DataFrame): labels

        Returns:
            BaselineOvO: Fitted estimator
        """
        self.models_ = {}
        for c1, c2 in combinations(self._rating_classes, 2):
            m = y.isin([c1, c2])
            Xp, yp = X.loc[m], y.loc[m]

            if self._minority_class in (c1, c2):
                Xp, yp = self._undersample(Xp, yp)

            pipe = Pipeline(
                [
                    ("preprocessing", build_logres_features(Xp)),
                    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )

            self.models_[(c1, c2)] = pipe.fit(Xp, yp)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class based on observation

        Args:
            X (pd.DataFrame): observation

        Returns:
            np.ndarray: The winning class and the probability assigned to it
        """
        n = len(X)
        classes = list(self._rating_classes)
        class_idx = {c: i for i, c in enumerate(classes)}

        votes = np.zeros((n, len(classes)))

        # Use probability as a tiebreaker
        proba_sum = np.zeros((n, len(classes)))

        for (c1, c2), pipe in self._models.items():
            pred = pipe.predict(X)
            for c in (c1, c2):
                votes[pred == c, class_idx[c]] += 1

            proba = pipe.predict_proba(X)
            for j, c in enumerate(pipe.named_steps["lr"].classes_):
                proba_sum[:, class_idx[c]] += proba[:, j]

        # Softmax
        exp = np.exp(proba_sum)
        probs = exp / exp.sum(axis=1, keepdims=True)

        # Rank first on # votes, then on probability
        ranked = np.lexsort((proba_sum, votes), axis=1)

        # lexsort goes from small to large, so reverse
        winners = ranked[:, -1]

        labels = np.array([classes[i] for i in winners])
        winner_conf = probs[np.arange(n), winners]
        return [
            (self._class_names[int(c)], round(float(proba), 3))
            for c, proba in zip(labels, winner_conf)
        ]
