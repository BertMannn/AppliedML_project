from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
import pandas as pd

from models.random_forest import RandomForestModel
from preprocessing.get_data import get_data


def eval_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = None,
    *,
    verbose: bool = False,
):
    X_train, X_test, y_train, y_test, numeric_columns = get_data(return_numeric_columns=True)

    rf_model = RandomForestModel(
        numeric_columns,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    rf_model.fit(X_train, y_train)

    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_uniform = DummyClassifier(strategy="uniform").fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    maj_dummy_pred = dummy_majority.predict(X_test)
    unif_dummy_pred = dummy_uniform.predict(X_test)

    models = {
        "Random Forest": y_pred_rf,
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
            print("\n")

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

    return rf_model, df_summary


if __name__ == "__main__":
    eval_random_forest(verbose=True)