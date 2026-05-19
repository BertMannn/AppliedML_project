from pathlib import Path

import pandas as pd
from feature_engineering import get_feature_engineering


CATEGORY_COLUMN = "RecipeCategory"
UNKNOWN_CATEGORY = "Unknown"


def encode_recipe_category() -> pd.DataFrame:
    """One-hot encode RecipeCategory and overwrite the features file."""
    df = get_feature_engineering()

    if CATEGORY_COLUMN not in df.columns:
        raise KeyError(f"Column '{CATEGORY_COLUMN}' was not found in features")

    missing_categories = df[CATEGORY_COLUMN].isna().sum()
    if missing_categories:
        print(
            f"Filling {missing_categories} missing RecipeCategory values with "
            f"'{UNKNOWN_CATEGORY}'..."
        )
        df[CATEGORY_COLUMN] = df[CATEGORY_COLUMN].fillna(UNKNOWN_CATEGORY)

    print("One-hot encoding RecipeCategory...")
    df = pd.get_dummies(
        df,
        columns=[CATEGORY_COLUMN],
        drop_first=True,
        dtype=int,
    )
    return df

def get_encoded_features():
    return encode_recipe_category()
