from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from preprocessing.text_features import BuildFeatureMatrix


MODEL_PATHS = {
    "random_forest": Path("models/random_forest.pkl"),
    "gradient_boosting": Path("models/gradient_boosting.pkl"),
}

CLASS_NAMES = {0: "<= 4.0", 1: "4.5", 2: "5.0"}

TEXT_COLUMNS = ("Name", "Description", "Keywords")
LIST_COLUMNS = ("Keywords", "RecipeIngredientParts", "RecipeInstructions")
NUMERIC_COLUMNS = (
    "RecipeServings",
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
)


class RecipeInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., alias="Name", min_length=1)
    description: str = Field("", alias="Description")
    recipe_category: str = Field("Unknown", alias="RecipeCategory")
    keywords: list[str] = Field(default_factory=list, alias="Keywords")
    recipe_ingredient_parts: list[str] = Field(
        default_factory=list, alias="RecipeIngredientParts"
    )
    recipe_instructions: list[str] = Field(
        default_factory=list, alias="RecipeInstructions"
    )
    cook_time: str | None = Field(None, alias="CookTime")
    prep_time: str | None = Field(None, alias="PrepTime")
    total_time: str | None = Field(None, alias="TotalTime")
    recipe_servings: float | None = Field(None, alias="RecipeServings", ge=0)
    calories: float | None = Field(None, alias="Calories", ge=0)
    fat_content: float | None = Field(None, alias="FatContent", ge=0)
    saturated_fat_content: float | None = Field(
        None, alias="SaturatedFatContent", ge=0
    )
    cholesterol_content: float | None = Field(None, alias="CholesterolContent", ge=0)
    sodium_content: float | None = Field(None, alias="SodiumContent", ge=0)
    carbohydrate_content: float | None = Field(
        None, alias="CarbohydrateContent", ge=0
    )
    fiber_content: float | None = Field(None, alias="FiberContent", ge=0)
    sugar_content: float | None = Field(None, alias="SugarContent", ge=0)
    protein_content: float | None = Field(None, alias="ProteinContent", ge=0)


class PredictionResponse(BaseModel):
    model: str
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float] | None = None


class BothPredictionResponse(BaseModel):
    random_forest: PredictionResponse
    gradient_boosting: PredictionResponse


app = FastAPI(
    title="Recipe Rating Classifier API",
    description=(
        "Predicts recipe rating classes from raw recipe features. "
        "Labels are <= 4.0, 4.5, and 5.0."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_feature_extractor() -> BuildFeatureMatrix:
    extractor = BuildFeatureMatrix()
    extractor.fit_transform(_training_frame_for_extractor())
    return extractor


@lru_cache(maxsize=1)
def get_numeric_columns() -> tuple[str, ...]:
    return tuple(get_feature_extractor().numeric_columns)


@lru_cache(maxsize=1)
def load_model(model_name: str):
    try:
        model_path = MODEL_PATHS[model_name]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown model.") from exc

    if not model_path.exists():
        raise HTTPException(
            status_code=500, detail=f"Model file not found: {model_path}"
        )

    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not load {model_name}: {exc}"
        ) from exc


def _training_frame_for_extractor() -> pd.DataFrame:
    from preprocessing.data_split import split_data

    train_df, _ = split_data()
    return train_df


def parse_duration_minutes(value: str | None) -> float | None:
    if not value:
        return None
    hours = re.search(r"(\d+)H", value)
    minutes = re.search(r"(\d+)M", value)
    if hours is None and minutes is None:
        raise HTTPException(
            status_code=422,
            detail=f"Duration '{value}' should look like PT20M or PT1H30M.",
        )
    hour_minutes = int(hours.group(1)) * 60 if hours else 0
    minute_minutes = int(minutes.group(1)) if minutes else 0
    return float(hour_minutes + minute_minutes)


def raw_recipe_to_feature_frame(recipe: RecipeInput) -> pd.DataFrame:
    data = recipe.model_dump(by_alias=True)

    row: dict[str, object] = {
        "Name": data["Name"],
        "Description": data["Description"],
        "Keywords": data["Keywords"],
        "RatingClass": 0,
        "CookTimeMinutes": parse_duration_minutes(data["CookTime"]),
        "PrepTimeMinutes": parse_duration_minutes(data["PrepTime"]),
        "TotalTimeMinutes": parse_duration_minutes(data["TotalTime"]),
        "KeywordCount": len(data["Keywords"]),
        "IngredientCount": len(data["RecipeIngredientParts"]),
        "InstructionStepCount": len(data["RecipeInstructions"]),
        "DescriptionLength": len(data["Description"]),
    }

    for column in NUMERIC_COLUMNS:
        row[column] = data[column]

    for column in get_numeric_columns():
        if column.startswith("RecipeCategory_"):
            row[column] = 0

    category_column = f"RecipeCategory_{data['RecipeCategory']}"
    if category_column in row:
        row[category_column] = 1

    for column in get_numeric_columns():
        row.setdefault(column, 0)

    ordered_columns = [*get_numeric_columns(), *TEXT_COLUMNS, "RatingClass"]
    return pd.DataFrame([{column: row.get(column, 0) for column in ordered_columns}])


def predict_with_model(
    model_name: Literal["random_forest", "gradient_boosting"],
    recipe: RecipeInput,
    *,
    include_probabilities: bool = True,
) -> PredictionResponse:
    model = load_model(model_name)
    extractor = get_feature_extractor()
    frame = raw_recipe_to_feature_frame(recipe)
    features, _ = extractor.transform(frame)

    if model_name == "gradient_boosting":
        labels, probabilities = model.predict(
            features,
            output_class_names=False,
            return_probability=include_probabilities,
        )
    else:
        labels, probabilities = model.predict(
            features,
            output_class_names=False,
            return_proba=include_probabilities,
        )

    predicted_class = int(labels[0])
    probability_payload = None
    if include_probabilities:
        probability_payload = {
            CLASS_NAMES[i]: round(float(probabilities[0][i]), 4)
            for i in range(len(probabilities[0]))
        }

    return PredictionResponse(
        model=model_name,
        predicted_class=predicted_class,
        predicted_label=CLASS_NAMES[predicted_class],
        probabilities=probability_payload,
    )


@app.get("/health", summary="Check whether the API is running")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict/random-forest",
    response_model=PredictionResponse,
    summary="Predict a recipe rating class with the Random Forest model",
)
def predict_random_forest(recipe: RecipeInput) -> PredictionResponse:
    return predict_with_model("random_forest", recipe)


@app.post(
    "/predict/gradient-boosting",
    response_model=PredictionResponse,
    summary="Predict a recipe rating class with the Gradient Boosting model",
)
def predict_gradient_boosting(recipe: RecipeInput) -> PredictionResponse:
    return predict_with_model("gradient_boosting", recipe)


@app.post(
    "/predict/both",
    response_model=BothPredictionResponse,
    summary="Predict a recipe rating class with both trained models",
)
def predict_both(recipe: RecipeInput) -> BothPredictionResponse:
    return BothPredictionResponse(
        random_forest=predict_with_model("random_forest", recipe),
        gradient_boosting=predict_with_model("gradient_boosting", recipe),
    )
