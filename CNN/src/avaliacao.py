from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential


class AvaliadorCNN:

    def __init__(
        self,
        modelo: Sequential,
        nomes_classes: list[str],
        diretorio_resultados: str = "resultados"
    ) -> None:

        self._modelo = modelo
        self._nomes_classes = nomes_classes

        self._diretorio_resultados = Path(diretorio_resultados)
        self._diretorio_resultados.mkdir(
            parents=True,
            exist_ok=True
        )

    def avaliar(
        self,
        imagens_teste: np.ndarray,
        rotulos_teste: np.ndarray
    ) -> dict:

        perda, acuracia = self._modelo.evaluate(
            imagens_teste,
            rotulos_teste,
            verbose=0
        )

        previsoes = self._obter_previsoes(
            imagens_teste
        )

        matriz_confusao = confusion_matrix(
            rotulos_teste,
            previsoes
        )

        relatorio = classification_report(
            rotulos_teste,
            previsoes,
            target_names=self._nomes_classes,
            digits=4
        )

        self._salvar_relatorio(
            perda,
            acuracia,
            matriz_confusao,
            relatorio
        )

        return {
            "perda": perda,
            "acuracia": acuracia,
            "matriz_confusao": matriz_confusao,
            "relatorio": relatorio
        }

    def _obter_previsoes(
        self,
        imagens: np.ndarray
    ) -> np.ndarray:

        probabilidades = self._modelo.predict(
            imagens,
            verbose=0
        )

        return np.argmax(
            probabilidades,
            axis=1
        )

    def _salvar_relatorio(
        self,
        perda: float,
        acuracia: float,
        matriz_confusao: np.ndarray,
        relatorio: str
    ) -> None:

        caminho = self._diretorio_resultados / "avaliacao.txt"

        with open(
            caminho,
            "w",
            encoding="utf-8"
        ) as arquivo:

            arquivo.write(
                f"Perda: {perda:.4f}\n"
            )

            arquivo.write(
                f"Acurácia: {acuracia:.4f}\n\n"
            )

            arquivo.write(
                "Matriz de Confusão\n"
            )

            arquivo.write(
                str(matriz_confusao)
            )

            arquivo.write("\n\n")

            arquivo.write(
                "Relatório de Classificação\n\n"
            )

            arquivo.write(relatorio)