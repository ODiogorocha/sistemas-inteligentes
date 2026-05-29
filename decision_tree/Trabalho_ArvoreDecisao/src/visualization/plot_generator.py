import os
import matplotlib.pyplot as plt


class PlotGenerator:

    @staticmethod
    def plot_top10(top10, path):

        plt.figure(figsize=(14, 8))

        plt.barh(
            top10["Categoria CID-10"],
            top10["Total"]
        )

        plt.xlabel("Quantidade de Óbitos")
        plt.ylabel("Categoria CID")
        plt.title("Top 10 Doenças com Maior Mortalidade")

        plt.tight_layout()

        plt.savefig(
            os.path.join(path, "top10_mortalidade.png"),
            dpi=300
        )

        plt.close()