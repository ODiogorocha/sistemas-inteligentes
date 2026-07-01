from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Graficos:

    def __init__(self):

        self.__diretorio_resultados = Path("resultados")

        self.__diretorio_resultados.mkdir(
            parents=True,
            exist_ok=True
        )

    def loss(self, historico):

        plt.figure(figsize=(10, 5))

        plt.plot(
            historico.history["loss"],
            label="Treinamento"
        )

        plt.plot(
            historico.history["val_loss"],
            label="Validação"
        )

        plt.title("Perda durante o treinamento")

        plt.xlabel("Épocas")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.__diretorio_resultados / "grafico_loss.png",
            dpi=300
        )

        plt.close()

    def previsao(
        self,
        valores_reais,
        valores_previstos
    ):

        valores_reais = np.asarray(
            valores_reais
        ).flatten()

        valores_previstos = np.asarray(
            valores_previstos
        ).flatten()

        plt.figure(figsize=(12, 6))

        plt.plot(
            valores_reais,
            label="Valores Reais"
        )

        plt.plot(
            valores_previstos,
            label="Valores Previstos"
        )

        plt.title("Valores Reais x Valores Previstos")

        plt.xlabel("Amostras")

        plt.ylabel("Passageiros")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.__diretorio_resultados / "grafico_previsao.png",
            dpi=300
        )

        plt.close()

    def serie_temporal(
        self,
        serie
    ):

        plt.figure(figsize=(12, 6))

        plt.plot(
            serie
        )

        plt.title("Série Temporal")

        plt.xlabel("Tempo")

        plt.ylabel("Passageiros")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.__diretorio_resultados / "grafico_serie_temporal.png",
            dpi=300
        )

        plt.close()

    def comparacao_completa(
        self,
        valores_reais,
        valores_previstos
    ):

        valores_reais = np.asarray(
            valores_reais
        ).flatten()

        valores_previstos = np.asarray(
            valores_previstos
        ).flatten()

        plt.figure(figsize=(14, 6))

        plt.plot(
            valores_reais,
            linewidth=2,
            label="Real"
        )

        plt.plot(
            valores_previstos,
            linewidth=2,
            linestyle="--",
            label="Previsto"
        )

        plt.fill_between(
            range(len(valores_reais)),
            valores_reais,
            valores_previstos,
            alpha=0.2
        )

        plt.title("Comparação Completa")

        plt.xlabel("Tempo")

        plt.ylabel("Passageiros")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.__diretorio_resultados / "comparacao_completa.png",
            dpi=300
        )

        plt.close()

    def dispersao(
        self,
        valores_reais,
        valores_previstos
    ):

        valores_reais = np.asarray(
            valores_reais
        ).flatten()

        valores_previstos = np.asarray(
            valores_previstos
        ).flatten()

        plt.figure(figsize=(8, 8))

        plt.scatter(
            valores_reais,
            valores_previstos
        )

        menor = min(
            valores_reais.min(),
            valores_previstos.min()
        )

        maior = max(
            valores_reais.max(),
            valores_previstos.max()
        )

        plt.plot(
            [menor, maior],
            [menor, maior]
        )

        plt.title("Dispersão")

        plt.xlabel("Valor Real")

        plt.ylabel("Valor Previsto")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            self.__diretorio_resultados / "grafico_dispersao.png",
            dpi=300
        )

        plt.close()