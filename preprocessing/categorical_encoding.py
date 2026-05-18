from pathlib import Path

import pandas as pd


FEATURES_PATH = Path("data/features.csv")
CATEGORY_COLUMN = "RecipeCategory"
UNKNOWN_CATEGORY = "Unknown"


def encode_recipe_category(data_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """One-hot encode RecipeCategory and overwrite the features file."""
    if data_path.suffix.lower() != ".csv":
        raise TypeError("The data should be a .csv file")

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist yet. Run feature_engineering.py first."
        )

    print(f"Reading features from {data_path}...")
    df = pd.read_csv(data_path)

    if CATEGORY_COLUMN not in df.columns:
        raise KeyError(f"Column '{CATEGORY_COLUMN}' was not found in {data_path}")

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

    print(f"Writing encoded features back to {data_path}...")
    df.to_csv(data_path, index=False)
    return df


def main():
    encode_recipe_category()


if __name__ == "__main__":
    main()
