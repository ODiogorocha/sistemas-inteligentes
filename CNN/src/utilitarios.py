from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class GeradorGraficos:

    def __init__(
        self,
        diretorio_resultados: str = "resultados"
    ) -> None:

        self._diretorio_resultados = Path(diretorio_resultados)
        self._diretorio_resultados.mkdir(
            parents=True,
            exist_ok=True
        )

    def gerar_graficos_treinamento(
        self,
        historico,
        nome_modelo: str
    ) -> None:

        self._gerar_grafico_acuracia(
            historico,
            nome_modelo
        )

        self._gerar_grafico_perda(
            historico,
            nome_modelo
        )

    def gerar_matriz_confusao(
        self,
        matriz_confusao: np.ndarray,
        nomes_classes: list[str],
        nome_modelo: str
    ) -> None:

        figura = plt.figure(figsize=(8, 6))

        plt.imshow(
            matriz_confusao,
            interpolation="nearest",
            cmap="Blues"
        )

        plt.title("Matriz de Confusão")

        plt.colorbar()

        posicoes = np.arange(len(nomes_classes))

        plt.xticks(
            posicoes,
            nomes_classes,
            rotation=45
        )

        plt.yticks(
            posicoes,
            nomes_classes
        )

        limite = matriz_confusao.max() / 2

        for linha in range(matriz_confusao.shape[0]):
            for coluna in range(matriz_confusao.shape[1]):

                plt.text(
                    coluna,
                    linha,
                    matriz_confusao[linha, coluna],
                    horizontalalignment="center",
                    color="white"
                    if matriz_confusao[linha, coluna] > limite
                    else "black"
                )

        plt.ylabel("Classe Real")
        plt.xlabel("Classe Predita")

        plt.tight_layout()

        caminho = (
            self._diretorio_resultados /
            f"{nome_modelo}_matriz_confusao.png"
        )

        figura.savefig(caminho)

        plt.close(figura)

    def gerar_grafico_comparacao(
        self,
        resultados: dict
    ) -> None:

        figura = plt.figure(figsize=(8, 5))

        nomes_modelos = list(resultados.keys())

        acuracias = [
            resultados[nome]
            for nome in nomes_modelos
        ]

        plt.bar(
            nomes_modelos,
            acuracias
        )

        plt.title("Comparação entre Modelos")

        plt.ylabel("Acurácia")

        plt.ylim(0, 1)

        caminho = (
            self._diretorio_resultados /
            "comparacao_modelos.png"
        )

        figura.savefig(caminho)

        plt.close(figura)

    def _gerar_grafico_acuracia(
        self,
        historico,
        nome_modelo: str
    ) -> None:

        figura = plt.figure(figsize=(8, 5))

        plt.plot(
            historico.history["accuracy"]
        )

        plt.plot(
            historico.history["val_accuracy"]
        )

        plt.title("Acurácia")

        plt.xlabel("Épocas")

        plt.ylabel("Acurácia")

        plt.legend(
            [
                "Treinamento",
                "Validação"
            ]
        )

        caminho = (
            self._diretorio_resultados /
            f"{nome_modelo}_acuracia.png"
        )

        figura.savefig(caminho)

        plt.close(figura)

    def _gerar_grafico_perda(
        self,
        historico,
        nome_modelo: str
    ) -> None:

        figura = plt.figure(figsize=(8, 5))

        plt.plot(
            historico.history["loss"]
        )

        plt.plot(
            historico.history["val_loss"]
        )

        plt.title("Perda")

        plt.xlabel("Épocas")

        plt.ylabel("Loss")

        plt.legend(
            [
                "Treinamento",
                "Validação"
            ]
        )

        caminho = (
            self._diretorio_resultados /
            f"{nome_modelo}_perda.png"
        )

        figura.savefig(caminho)

        plt.close(figura)