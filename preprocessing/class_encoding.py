import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("data/clean_recipe_data.csv"))


df["AggregatedRating"] = df["AggregatedRating"].astype(float)

# Map the values to the three rating classes:
df["RatingClass"] = df["AggregatedRating"].apply(
    lambda x: 2 if x == 5.0 else (1 if x == 4.5 else 0)
)
