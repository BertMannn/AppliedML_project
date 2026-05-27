# AppliedML_project

FastAPI deployment for the recipe rating classification models.

The API accepts raw recipe features, applies the same preprocessing used during
training, and returns human-readable rating classes:

- `<= 4.0`
- `4.5`
- `5.0`

## Models

The repository contains two trained models:

- `models/random_forest.pkl`
- `models/gradient_boosting.pkl`

On the validation/test split used in this project:

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 0.6706 | 0.3913 | 0.6559 |
| Gradient Boosting | 0.5961 | 0.4059 | 0.6233 |
| Dummy uniform/random | 0.3317 | 0.2682 | - |
| Dummy majority | 0.7507 | 0.2859 | - |

Random Forest has higher accuracy. Gradient Boosting has higher macro F1, which
is useful because the classes are imbalanced.

## Installation

Install the Python dependencies:

```bash
uv sync
```

LightGBM may require OpenMP on macOS:

```bash
brew install libomp
```

## Run The API

Start the server:

```bash
uv run uvicorn api:app --reload
```

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Endpoints

- `GET /health`
- `POST /predict/random-forest`
- `POST /predict/gradient-boosting`
- `POST /predict/both`

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict/both" \
  -H "Content-Type: application/json" \
  -d '{
    "Name": "Best Lemonade",
    "Description": "Fresh lemon drink with sugar and chilled water.",
    "RecipeCategory": "Beverages",
    "Keywords": ["Low Protein", "Healthy", "Summer", "< 60 Mins"],
    "RecipeIngredientParts": ["sugar", "lemons", "water", "fresh lemon juice"],
    "RecipeInstructions": ["Mix sugar and lemon peel.", "Add lemon juice.", "Serve chilled."],
    "CookTime": "PT5M",
    "PrepTime": "PT30M",
    "TotalTime": "PT35M",
    "RecipeServings": 4,
    "Calories": 311.1,
    "FatContent": 0.2,
    "SaturatedFatContent": 0,
    "CholesterolContent": 0,
    "SodiumContent": 1.8,
    "CarbohydrateContent": 81.5,
    "FiberContent": 0.4,
    "SugarContent": 77.2,
    "ProteinContent": 0.3
  }'
```

Example response:

```json
{
  "random_forest": {
    "model": "random_forest",
    "predicted_class": 2,
    "predicted_label": "5.0",
    "probabilities": {
      "<= 4.0": 0.2393,
      "4.5": 0.2979,
      "5.0": 0.4628
    }
  },
  "gradient_boosting": {
    "model": "gradient_boosting",
    "predicted_class": 2,
    "predicted_label": "5.0",
    "probabilities": {
      "<= 4.0": 0.2248,
      "4.5": 0.1796,
      "5.0": 0.5956
    }
  }
}
```

## Submission Evidence

For the model deployment assignment, collect:

- A screenshot of `http://127.0.0.1:8000/docs`.
- A screenshot or terminal capture of a successful `curl` or Postman request.
- The RF vs GBM comparison table above.
- A short note that both models beat random guessing on macro F1.
