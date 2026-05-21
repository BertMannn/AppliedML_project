from sklearn.metrics import f1_score, confusion_matrix, classification_report

from models.baseline import BaselineOvO
from preprocessing.get_data import get_data


def eval_baseline():
    X_train, X_test, y_train, y_test, numeric_columns = get_data(
        return_numeric_columns=True
    )
    baseline_logres = BaselineOvO(numeric_columns, max_iter=1000)
    baseline_logres.fit(X_train, y_train)

    y_pred = baseline_logres.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    clf_rpt = classification_report(y_test, y_pred, digits=4)

    print(f"Macro-F1: {macro_f1:.4f}")
    print(clf_rpt)
    print("Confusion matrix:")
    print(cm)

    return macro_f1, cm


if __name__ == "__main__":
    eval_baseline()
