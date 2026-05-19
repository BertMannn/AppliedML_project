from pathlib import Path

import pandas as pd

from data_split import split_data
from text_features import BuildFeatureMatrix


def get_data(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, ...]:
    """Get all the formatted data. Standard formatted with an 80/20 train/test split

    Args:
        test_size (float, optional): Percentage of test data. Defaults to 0.2.
        random_state (int, optional): random state. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, ...]: Train test split as DataFrames
    """

    train_df, test_df = split_data(
        test_size=test_size, random_state=random_state
    )

    feature_extractor = BuildFeatureMatrix()
    X_train, y_train = feature_extractor.fit_transform(train_df)
    X_test, y_test = feature_extractor.transform(test_df)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test: ", y_test.shape)
    print("y_train counts:\n", y_train.value_counts())
    print("y_test counts:\n", y_test.value_counts())