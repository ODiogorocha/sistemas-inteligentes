from pathlib import Path

import numpy as np
import pandas as pd


class CarregadorDados:

    def __init__(self, caminho_arquivo):
        self.__caminho_arquivo = Path(caminho_arquivo)
        self.__dados = None

    @property
    def dados(self):
        return self.__dados

    def carregar(self):
        self.__validar_arquivo()
        self.__dados = pd.read_csv(self.__caminho_arquivo)
        self.__validar_colunas()
        return self.__dados

    def obter_serie_temporal(self):
        if self.__dados is None:
            raise RuntimeError("Os dados ainda não foram carregados.")

        return self.__dados["Passengers"].values.astype(np.float32)

    def obter_datas(self):
        if self.__dados is None:
            raise RuntimeError("Os dados ainda não foram carregados.")

        return self.__dados["Month"]

    def quantidade_registros(self):
        if self.__dados is None:
            return 0

        return len(self.__dados)

    def quantidade_colunas(self):
        if self.__dados is None:
            return 0
        return len(self.__dados.columns)

    def exibir_resumo(self):
        if self.__dados is None:
            raise RuntimeError("Os dados ainda não foram carregados.")

        print()
        print("=" * 50)
        print("RESUMO DO DATASET")
        print("=" * 50)
        print(f"Registros : {self.quantidade_registros()}")
        print(f"Colunas   : {self.quantidade_colunas()}")
        print()
        print(self.__dados.head())
        print()

    def __validar_arquivo(self):
        if not self.__caminho_arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.__caminho_arquivo}"
            )

    def __validar_colunas(self):

        colunas = list(self.__dados.columns)
        esperado = [
            "Month",
            "Passengers"
        ]

        if colunas != esperado:
            raise ValueError(
                "O dataset deve possuir as colunas Month e Passengers."
            )