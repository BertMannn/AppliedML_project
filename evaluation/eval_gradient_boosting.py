from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from models.gradient_boosting import GradientBoostingModel
from preprocessing.get_data import get_data
from models.feature_prep import build_features


def eval_gradient_boosting(*, verbose: bool = False):
    X_train, X_test, y_train, y_test, numeric_columns = get_data(
        return_numeric_columns=True
    )
    #  Hyperparameter optimisation via GridSearchCV:

    pipeline = Pipeline(
        [
            ("preprocessing", build_features(numeric_columns)),
            (
                "gbm",
                LGBMClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )

    parameter_grid = {
        "gbm__learning_rate": [0.05, 0.1],
        "gbm__n_estimators": [200, 500],
        "gbm__max_depth": [5, 10],
    }

    grid_search = GridSearchCV(
        pipeline,
        parameter_grid,
        scoring=make_scorer(f1_score, average="macro"),
        cv=5,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_index = grid_search.best_index_
    cv_std = grid_search.cv_results_["std_test_score"][best_index]

    print(f"Best CV macro F1: {grid_search.best_score_:.4f} ± {cv_std:.4f}")
    print(f"\nBest parameters:   {grid_search.best_params_}")

    #  Over/underfitting evaluation: 

    train_score = f1_score(
        y_train, grid_search.best_estimator_.predict(X_train), average="macro"
    )
    val_score = grid_search.best_score_

    print(f"\nTrain macro F1: {train_score:.4f}")
    print(f"CV validation macro F1: {val_score:.4f}")
    print(f"Gap (train - val): {train_score - val_score:.4f}")

    if train_score - val_score > 0.1:
        print("Potential overfitting detected")
    elif train_score < 0.35:
        print("Potential underfitting detected")
    else:
        print("Fit looks reasonable.")

    #  Final evaluation on test set: 

    best_gbm = GradientBoostingModel(
        numeric_columns,
        learning_rate=grid_search.best_params_["gbm__learning_rate"],
        n_estimators=grid_search.best_params_["gbm__n_estimators"],
        max_depth=grid_search.best_params_["gbm__max_depth"],
    )
    best_gbm.fit(X_train, y_train)

    joblib.dump(best_gbm, "models/gradient_boosting.pkl")
    print("Model saved to models/gradient_boosting.pkl")

    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_uniform = DummyClassifier(strategy="uniform").fit(X_train, y_train)

    y_pred_gbm = best_gbm.predict(X_test)
    maj_dummy_pred = dummy_majority.predict(X_test)
    unif_dummy_pred = dummy_uniform.predict(X_test)

    models = {
        "Gradient Boosting (best)": y_pred_gbm,
        "Dummy (Majority)": maj_dummy_pred,
        "Dummy (Uniform)": unif_dummy_pred,
    }

    summary_stats = []

    for name, y_pred in models.items():
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report_dict["accuracy"]

        if verbose:
            print("\n==================================================")
            print(f" MODEL: {name}")
            print("==================================================")
            print(classification_report(y_test, y_pred, digits=4))
            print("Confusion matrix:")
            print(cm)

        summary_stats.append(
            {
                "model_name": name,
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "accuracy": round(accuracy, 4),
            }
        )

    df_summary = pd.DataFrame(summary_stats)

    print("\n" + "=" * 50)
    print("       GRADIENT BOOSTING MODEL COMPARISON          ")
    print("=" * 50)
    print(df_summary.to_string(index=False))
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_gbm, display_labels=["≤4.0", "4.5", "5.0"], ax=ax
    )
    plt.tight_layout()
    plt.savefig("evaluation/confusion_matrix_gbm.png", dpi=150)
    plt.close()

    return best_gbm, df_summary



def eval_saved_gradient_boosting(
    *, model_path: str = "models/gradient_boosting.pkl", verbose: bool = False
):
    """Evaluate the previously trained gradient boosting model on the test set.
 
    Loads the pickled best model from `model_path` and compares it against
    majority- and uniform-class dummy baselines. Skips hyperparameter search.
    """
    X_train, X_test, y_train, y_test, _ = get_data(return_numeric_columns=True)
 
    best_gbm = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
 
    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_uniform = DummyClassifier(strategy="uniform").fit(X_train, y_train)
 
    y_pred_gbm = best_gbm.predict(X_test)
    maj_dummy_pred = dummy_majority.predict(X_test)
    unif_dummy_pred = dummy_uniform.predict(X_test)
 
    models = {
        "Gradient Boosting (best)": y_pred_gbm,
        "Dummy (Majority)": maj_dummy_pred,
        "Dummy (Uniform)": unif_dummy_pred,
    }
 
    summary_stats = []
 
    for name, y_pred in models.items():
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report_dict["accuracy"]
 
        if verbose:
            print("\n==================================================")
            print(f" MODEL: {name}")
            print("==================================================")
            print(classification_report(y_test, y_pred, digits=4))
            print("Confusion matrix:")
            print(cm)
 
        summary_stats.append(
            {
                "model_name": name,
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "accuracy": round(accuracy, 4),
            }
        )
 
    df_summary = pd.DataFrame(summary_stats)
 
    print("\n" + "=" * 50)
    print("       GRADIENT BOOSTING MODEL COMPARISON          ")
    print("=" * 50)
    print(df_summary.to_string(index=False))
    print("=" * 50)
 
    return best_gbm, df_summary


if __name__ == "__main__":
    # eval_gradient_boosting(verbose=True)
    eval_saved_gradient_boosting(verbose=True)