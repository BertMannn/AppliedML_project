import pandas as pd
from pathlib import Path
import re


class CleanRecipeData:
    def __call__(self, data_path: Path, *, write_new_data: bool = False):
        """When you call the class the data will be cleaned. This is done by
        removing useless columns, NA values, and rows with reviewcount <10

        Args:
            data_path (Path): Path to the data
            write_new_data (bool, optional): If you want to overwrite the existing clean data. Defaults to False.

        Raises:
            TypeError: The path should lead to a .csv file.
        """
        if data_path.suffix.lower() != ".csv":
            err_msg = "The data should be a .csv file"
            raise TypeError(err_msg)

        print("Reading data...")
        df = pd.read_csv(data_path)

        print(f"Rows before cleaning: {len(df)}")

        print("Dropping Duplicates...")
        df = df.drop_duplicates()

        print("Dropping NA or <5 reviews...")
        df = df[df["ReviewCount"] != "NA"]
        df = df.dropna(subset=["ReviewCount"])
        print(f"Rows after dropping NA: {len(df)}")

        df = df[df["ReviewCount"].astype(int) >= 5]
        print(f"Rows after dropping ReviewCount <10: {len(df)}")

        print("Dropping useless columns...")
        useless_columns = [
            "RecipeId",
            "AuthorId",
            "AuthorName",
            "Images",
            "RecipeYield",
            "DatePublished",
            "RecipeIngredientQuantities",
        ]
        df = df.drop(useless_columns, axis=1)

        print(f"Rows after cleaning: {len(df)}")

        vector_cols = ["Keywords", "RecipeIngredientParts", "RecipeInstructions"]

        for col in vector_cols:
            df[col] = df[col].apply(
                lambda x: self._convert_r_vectors(x) if isinstance(x, str) else x
            )   


        if write_new_data:
            self._write_clean_data(df)

    def _write_clean_data(self, clean_data: pd.DataFrame):
        """Write the clean data to a new csv

        Args:
            clean_data (pd.DataFrame): The cleaned data
        """
        print("Writing new data to /data/clean_recipe_data.csv")
        output_path = Path(__file__).parent.parent / "data" / "clean_recipe_data.csv"
        clean_data.to_csv(output_path, index=False)

    def _convert_r_vectors(self, r_vec: str) -> list[str]:
        """Original data was meant to be used for R, but we are
        using Python, so we need to convert R vectors to
        python lists

        Args:
            r_vec (str): R vector to convert to list

        Returns:
            list[str]: List containing the content of the
            R vector, the content is all strings.
        """
        r_vec = r_vec.strip()

        # R vectors start with "c("
        if r_vec.startswith("c("):
            vector_content = r_vec[2:-1]
            # Find all items in the vector
            items = re.findall(r'"((?:[^"\\]|\\.)*)"', vector_content)
            return items


def main():
    """Original Recipe data can be found here:
    https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews?resource=download
    Did not upload the dataset to GitHub, since it is rather large.
    """
    data_path = Path("data/recipes.csv")
    CleanRecipeData()(data_path, write_new_data=True)


if __name__ == "__main__":
    main()
