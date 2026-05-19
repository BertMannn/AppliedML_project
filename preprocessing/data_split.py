import pandas as pd
from sklearn.model_selection import train_test_split
from categorical_encoding import get_encoded_features


def split_data(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    target: str = "RatingClass",
) -> pd.DataFrame:
    """Get an 80/20 data split and load it into memory. In get_data.py this
    is done before vectorization to prevent leakage.

    Args:
        test_size (float, optional): percentage of test data. Defaults to 0.2.
        random_state (int, optional): random state. Defaults to 42.
        target (str, optional): The label target we want to predict. Defaults to "RatingClass".

    Raises:
        TypeError: We read csv, so the file should be a csv

    Returns:
        pd.DataFrame: The train and test dataframes
    """
    df = get_encoded_features()

    # Stratify on the target so the class balance is preserved in both splits.
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )

    print(f"Split: {len(train_df)} train rows, {len(test_df)} test rows")
    return train_df, test_df
