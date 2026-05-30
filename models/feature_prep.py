from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


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


def build_features(numeric_block_columns: list[str]) -> ColumnTransformer:
    """Build preprocessing ColumnTransformer for sparse matrix

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

