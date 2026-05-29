class DescriptiveAnalysis:

    @staticmethod
    def show_general_info(df):

        print("\nInformações gerais:")
        print(df.info())

        print("\nValores nulos:")
        print(df.isnull().sum())

        print("\nEstatísticas descritivas:")
        print(df.describe())

    @staticmethod
    def show_top_diseases(df):

        top10 = (
            df[["Categoria CID-10", "Total"]]
            .sort_values(by="Total", ascending=False)
            .head(10)
        )

        print("\nTop 10 doenças com maior mortalidade:\n")
        print(top10)

        return top10