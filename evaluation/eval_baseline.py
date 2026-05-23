from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.dummy import DummyClassifier

from models.baseline import BaselineLR
from preprocessing.get_data import get_data

def eval_baseline(max_iter: int = 1000, *, verbose: bool = False):
    X_train, X_test, y_train, y_test, numeric_columns = get_data(
        return_numeric_columns=True
    )

    baseline_logres = BaselineLR(numeric_columns, max_iter=max_iter)

    # Fit our own model
    baseline_logres.fit(X_train, y_train)

    # Fit standard sklearn models
    baseline_logres.fit_models(X_train, y_train)

    dummy_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_uniform = DummyClassifier(strategy="uniform").fit(X_train, y_train)

    # Predict own model
    y_pred_baseline = baseline_logres.predict(X_test)

    # Predict standard sklearn models
    y_pred_std, y_pred_ovr, y_pred_ovo = baseline_logres.predict_all(
        X_test
    )

    # Show that predictions are not random
    maj_dummy_pred = dummy_majority.predict(X_test)
    unif_dummy_pred = dummy_uniform.predict(X_test)



    models = {
        "Baseline": y_pred_baseline,
        "Default (Standard)": y_pred_std,
        "Default (OvR)": y_pred_ovr,
        "Default (OvO)": y_pred_ovo,
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
            print("confusion matrix:")
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
    print("             FINAL MODEL COMPARISON               ")
    print("=" * 50)
    print(df_summary.to_string(index=False))
    print("=" * 50)

    return models, df_summary

if __name__ == "__main__":
    eval_baseline()
