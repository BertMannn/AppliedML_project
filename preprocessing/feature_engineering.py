import pandas as pd
from pathlib import Path
import re
import ast


df = pd.read_csv(Path("data/clean_recipe_data.csv"))

vector_cols = ["Keywords", "RecipeIngredientParts", "RecipeInstructions"]
for col in vector_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

df["AggregatedRating"] = df["AggregatedRating"].astype(float)

df["RatingClass"] = df["AggregatedRating"].apply(
    lambda x: 2 if x == 5.0 else (1 if x == 4.5 else 0)
)

# -------------------
# Feature engineering
# -------------------

# Since the original preptime, cooktime and totaltime values are formatted as 'PT20M' for example,
# we want to turn these values into integers:
def parse_duration(val):
    """Turn the duration values to minutes."""
    if pd.isna(val) or not isinstance(val, str):
        return None
    hours = re.search(r'(\d+)H', val)
    minutes = re.search(r'(\d+)M', val)
    return (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)

df["CookTimeMinutes"] = df["CookTime"].apply(parse_duration)
df["PrepTimeMinutes"] = df["PrepTime"].apply(parse_duration)
df["TotalTimeMinutes"] = df["TotalTime"].apply(parse_duration)


# Make sure other numeric values are floats:
numeric_cols = ["RecipeServings", "Calories", "FatContent", "SaturatedFatContent",
                "CholesterolContent", "SodiumContent", "CarbohydrateContent",
                "FiberContent", "SugarContent", "ProteinContent"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create count features from text features, for example to be able to compare the length of the
# instructions:
def safe_len(val):
    return len(val) if isinstance(val, list) else 0
# This prevents crashing in case a value is not a list

df["KeywordCount"] = df["Keywords"].apply(safe_len)
df["IngredientCount"] = df["RecipeIngredientParts"].apply(safe_len)
df["InstructionStepCount"] = df["RecipeInstructions"].apply(safe_len)

if "Description" in df.columns:
    df["DescriptionLength"] = df["Description"].apply(
        lambda x: len(x) if isinstance(x, str) else 0
    )

# Save features:
output_path = Path("data/features.csv")
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")