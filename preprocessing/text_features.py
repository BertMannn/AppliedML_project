import ast

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class BuildFeatureMatrix:
    """Build sparse feature matrix from recipe data

    First use fit_transform to create vectorizers and transform training data.
    Then use transform to transform testing data.

        Attributes:
        TEXT_COLUMNS: Text columns converted into numeric features via TF-IDF.
        DROP_COLUMNS: Columns removed before building the matrix because they
            are uninformative or could leak the target.
        MAX_NUMBER_FEATURES_PER_COLUMN: Vocabulary cap per text column.
        TARGET: Name of the label column.
    """

    TEXT_COLUMNS: tuple[str, ...] = ("Name", "Description", "Keywords")

    DROP_COLUMNS: tuple[str, ...] = (
        "CookTime",
        "PrepTime",
        "TotalTime",
        "AggregatedRating",
        "RecipeIngredientParts",
        "RecipeInstructions",
    )

    MAX_NUMBER_FEATURES_PER_COLUMN: int = 750

    TARGET: str = "RatingClass"

    def __init__(self) -> None:
        """Initialize with an empty vectorizer store (nothing fitted yet)."""
        self._vectorizers: dict[str, TfidfVectorizer] = {}

    def fit_transform(self, df: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.Series]:
        """Fit vectorizers on the training data, fits one vectorizer per text column

        Args:
            df (pd.DataFrame): Training df

        Returns:
            tuple[sparse.csr_matrix, pd.Series]: X: Sparse CSR matrix with transformed features, y: Labels
        """
        X, y = self._prepare_dataframe(df)

        print("Fitting + transforming text columns with TF-IDF...")
        tfidf_matrices: list[sparse.csr_matrix] = []
        for col in self.TEXT_COLUMNS:
            vectorizer = TfidfVectorizer(
                max_features=self.MAX_NUMBER_FEATURES_PER_COLUMN,
                # Drop common stop words that don't affect the rating.
                stop_words="english",
                lowercase=True,
            )
            matrix = vectorizer.fit_transform(X[col])
            self._vectorizers[col] = vectorizer
            print(f"  {col}: {matrix.shape[1]} features")
            tfidf_matrices.append(matrix)

        return self._combine_dataframes(X, tfidf_matrices), y

    def transform(self, df: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.Series]:
        """Transform test data using fitted vectorizers

        Args:
            df (pd.DataFrame): df to transform

        Raises:
            RuntimeError: Should fit vectorizers before transforming.

        Returns:
            tuple[sparse.csr_matrix, pd.Series]: X: Sparse CSR matrix with transformed features, y: Labels
        """

        if not self._vectorizers:
            raise RuntimeError("Call fit_transform on the training data first.")

        X, y = self._prepare_dataframe(df)

        print("Transforming text columns with TF-IDF...")
        tfidf_matrices: list[sparse.csr_matrix] = []
        for col in self.TEXT_COLUMNS:
            vectorizer = self._vectorizers[col]
            tfidf_matrices.append(vectorizer.transform(X[col]))

        return self._combine_dataframes(X, tfidf_matrices), y

    def _prepare_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split the target from the df, remove columns we dont want, and clean text columns

        Args:
            df (pd.DataFrame): Raw input df

        Returns:
            tuple[pd.DataFrame, pd.Series]: Feature df and target as pd.Series
        """
        df = df.copy()

        y = df[self.TARGET]
        df = df.drop(columns=[self.TARGET])
        df = df.drop(columns=list(self.DROP_COLUMNS), errors="ignore")

        # Keywords is a stringified list -> flatten to plain text for TF-IDF.
        df["Keywords"] = df["Keywords"].apply(self._flatten_keyword_list)
        for col in self.TEXT_COLUMNS:
            df[col] = df[col].fillna("")

        return df, y

    def _combine_dataframes(
        self, X: pd.DataFrame, tfidf_matrices: list[sparse.csr_matrix]
    ) -> sparse.csr_matrix:
        """Stack all the data into one sparse CSR matrix

        Args:
            X (pd.DataFrame): Feature names returned by prepare
            tfidf_matrices (list[sparse.csr_matrix]): The TFIDF matrices, one for each column in TEXT_COLUMNS

        Returns:
            sparse.csr_matrix: The combined feature matrix as a sparse CSR matrix
        """

        numeric_df = X.drop(columns=list(self.TEXT_COLUMNS)).fillna(0)

        # Numeric block as a sparse matrix, then stack everything horizontally.
        numeric_matrix = sparse.csr_matrix(numeric_df.values.astype(float))
        combined = sparse.hstack([numeric_matrix, *tfidf_matrices], format="csr")
        return combined

    def _flatten_keyword_list(self, raw: str) -> str:
        """Turn a list of keywords contraining strings into space-separeted text

        Args:
            raw (str): A values from the keywords column

        Returns:
            str: The keywords, but space-separated instead of string
        """

        if not isinstance(raw, str):
            return ""
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed)
        return str(parsed)
