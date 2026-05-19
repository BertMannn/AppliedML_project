from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    data_path: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    target: str = "RatingClass",
) -> pd.DataFrame:
    """Get an 80/20 data split and load it into memory. In get_data.py this
    is done before vectorization to prevent leakage.

    Args:
        data_path (Path): The path to the features
        test_size (float, optional): percentage of test data. Defaults to 0.2.
        random_state (int, optional): random state. Defaults to 42.
        target (str, optional): The label target we want to predict. Defaults to "RatingClass".

    Raises:
        TypeError: We read csv, so the file should be a csv

    Returns:
        pd.DataFrame: The train and test dataframes
    """
    if data_path.suffix.lower() != ".csv":
        raise TypeError("The data should be a .csv file")

    df = pd.read_csv(data_path)

    # Stratify on the target so the class balance is preserved in both splits.
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )

    print(f"Split: {len(train_df)} train rows, {len(test_df)} test rows")
    return train_df, test_df
