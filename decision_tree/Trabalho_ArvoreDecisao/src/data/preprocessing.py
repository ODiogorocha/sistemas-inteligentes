import pandas as pd
import numpy as np


class Preprocessing:

    @staticmethod
    def clean_data(df):

        for col in df.columns[1:]:

            df[col] = (
                df[col]
                .astype(str)
                .str.replace("-", "0")
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.fillna(0, inplace=True)

        return df

    @staticmethod
    def create_target(df):

        mediana = df["Total"].median()

        df["high_mortality"] = np.where(
            df["Total"] > mediana,
            1,
            0
        )

        return df