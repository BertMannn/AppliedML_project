import ast
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

class ExtractingTextFeatures:
    #text columns that are changed into numeric features using TF-IDF.
    TEXT_COLUMNS: tuple[str, ...] = ("Name", "Description", "Keywords")

    #dropping the following columns before building the feature matrix because they are not useful
    #and could lead to data leakage.
    DROP_COLUMNS: tuple[str, ...] = (
        "CookTime",
        "PrepTime",
        "TotalTime",
        "AggregatedRating",
        "RecipeIngredientParts",
        "RecipeInstructions",
    )

    #max number of words that is seen as a feature per text column
    MAX_NUMBER_FEATURES_PER_COLUMN: int = 750

    #vectorizing text columns using TF-IDF and combine it with numeric features.
    def __call__(self, data_path: Path, *, write_new_data: bool = False):
        if data_path.suffix.lower() != ".csv":
            raise TypeError("The data should be a .csv file")

        print("Reading data...")
        df = pd.read_csv(data_path)
        print(f"Rows loaded: {len(df)}, columns: {df.shape[1]}")

        #removing target variable before creating features, so we do not use it as a feature.
        print("Separating target variable (RatingClass)...")
        y = df["RatingClass"]
        df = df.drop(columns=["RatingClass"])

        #dropping all columns that we cannot use.
        print(f"Dropping columns: {self.DROP_COLUMNS}")
        df = df.drop(columns=list(self.DROP_COLUMNS), errors="ignore")

        #for TF-IDF we want a flat string. The vectorizer can then count the individual words as tokens.
        print("Flattening Keywords list-strings into plain text...")
        df["Keywords"] = df["Keywords"].apply(self._flatten_keyword_list)

        for col in self.TEXT_COLUMNS:
            df[col] = df[col].fillna("")

        #Each text column gets its own TF-IDF matrix. They are vectorized seperately.
        print("Vectorizing text columns with TF-IDF...")
        tfidf_list = []
        for col in self.TEXT_COLUMNS:
            matrix = self._vectorize_column(df[col], col)
            tfidf_list.append(matrix)

        numeric_df = df.drop(columns=list(self.TEXT_COLUMNS))
        print(f"Numeric/encoded feature columns: {numeric_df.shape[1]}")

        #numeric dataframe is transformed to a sparse matrix.
        numeric_matrix = sparse.csr_matrix(
            numeric_df.fillna(0).values.astype(float)
        )

        print("Combining numeric features and TF-IDF matrices...")
        X = sparse.hstack([numeric_matrix, *tfidf_list], format="csr")
        print(f"Final feature matrix shape: {X.shape}")

        if write_new_data:
            self._write_features(X, y)

    #stringified list of keywords is converted into a plain string.
    def _flatten_keyword_list(self, raw: str) -> str:

        if not isinstance(raw, str):
            return ""
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)
        return str(parsed)

    #apply a TF-IDF vectorizer on a text column and return its matrix.
    def _vectorize_column(self, series: pd.Series, column_name: str):

        vectorizer = TfidfVectorizer(
            max_features=self.MAX_NUMBER_FEATURES_PER_COLUMN,
            #removing stop words like "the", "a" and "is", because they do not influence the rating
            stop_words="english",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(series)
        print(f"  {column_name}: {matrix.shape[1]} features")
        return matrix

    #saving the feature matrix and then labeling vector to data.
    def _write_features(self, X, y: pd.Series):

        data_dir = Path(__file__).parent.parent / "data"

        x_path = data_dir / "X_features.npz"
        y_path = data_dir / "y_labels.csv"

        print(f"Writing feature matrix to {x_path}")
        sparse.save_npz(x_path, X)

        print(f"Writing labels to {y_path}")
        y.to_csv(y_path, index=False)


def main():

    data_path = Path("data/features.csv")
    ExtractingTextFeatures()(data_path, write_new_data=True)


if __name__ == "__main__":
    main()