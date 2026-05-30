from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

from models.random_forest import RandomForestModel
from preprocessing.get_data import get_data
from models.feature_prep import build_features


def eval_random_forest(*, verbose: bool = False):
    X_train, X_test, y_train, y_test, numeric_columns = get_data(
        return_numeric_columns=True
    )
    # ---------------------------------------------- #
    #  Hyperparameter optimisation via GridSearchCV: #
    # ---------------------------------------------- #

    pipeline = Pipeline(
        [
            ("preprocessing", build_features(numeric_columns)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "rf__max_depth": [10, 15, 20],
        "rf__min_samples_leaf": [3, 5, 10],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=make_scorer(f1_score, average="macro"),
        cv=5,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest params:   {grid_search.best_params_}")
    print(f"Best CV macro F1: {grid_search.best_score_:.4f}")

    # ------------------------------ #
    #  Over/underfitting evaluation: #
    # ------------------------------ #

    train_score = f1_score(
        y_train, grid_search.best_estimator_.predict(X_train), average="macro"
    )
    val_score = grid_search.best_score_

    print(f"\nTrain macro F1: {train_score:.4f}")
    print(f"CV val macro F1: {val_score:.4f}")
    print(f"Gap (train - val): {train_score - val_score:.4f}")

    if train_score - val_score > 0.1:
        print("Potential overfitting detected")
    elif train_score < 0.35:
        print("Potential underfitting detected")
    else:
        print("Fit looks reasonable.")

    # ------------------------------ #
    #  Final evaluation on test set: #
    # ------------------------------ #
    best_rf = RandomForestModel(
        numeric_columns,
        n_estimators=200,
        max_depth=grid_search.best_params_["rf__max_depth"],
        min_samples_leaf=grid_search.best_params_["rf__min_samples_leaf"],
    )
    best_rf.fit(X_train, y_train)

    joblib.dump(best_rf, "models/random_forest.pkl")
    print("Model saved to models/random_forest.pkl")

    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_uniform = DummyClassifier(strategy="uniform").fit(X_train, y_train)

    y_pred_rf = best_rf.predict(X_test)
    maj_dummy_pred = dummy_majority.predict(X_test)
    unif_dummy_pred = dummy_uniform.predict(X_test)

    models = {
        "Random Forest (best)": y_pred_rf,
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
    print("         RANDOM FOREST MODEL COMPARISON           ")
    print("=" * 50)
    print(df_summary.to_string(index=False))
    print("=" * 50)

    return best_rf, df_summary


if __name__ == "__main__":
    eval_random_forest(verbose=True)