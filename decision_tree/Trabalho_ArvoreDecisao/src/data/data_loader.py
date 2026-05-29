import pandas as pd


class DataLoader:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):

        df = pd.read_csv(
            self.dataset_path,
            sep=";",
            encoding="latin1",
            skiprows=3,
            engine="python"
        )

        df = df.dropna(axis=1, how="all")
        df = df.dropna(how="all")

        df.columns = [col.strip() for col in df.columns]

        df = df[
            ~df.iloc[:, 0].astype(str).str.contains(
                "Fonte|Notas|Período",
                case=False,
                na=False
            )
        ]

        df.reset_index(drop=True, inplace=True)

        return df