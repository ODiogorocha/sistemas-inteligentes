import numpy as np


class GeradorSequencias:

    def __init__(
        self,
        tamanho_janela=12,
        percentual_treino=0.80
    ):

        self.__tamanho_janela = tamanho_janela
        self.__percentual_treino = percentual_treino

    @property
    def tamanho_janela(self):

        return self.__tamanho_janela

    @property
    def percentual_treino(self):

        return self.__percentual_treino

    def gerar(self, serie):

        x, y = self.__criar_sequencias(serie)

        return self.__dividir_treino_teste(x, y)

    def __criar_sequencias(self, serie):

        entradas = []
        saidas = []

        for indice in range(
            self.__tamanho_janela,
            len(serie)
        ):

            entradas.append(
                serie[
                    indice - self.__tamanho_janela:indice,
                    0
                ]
            )

            saidas.append(
                serie[indice, 0]
            )

        entradas = np.array(
            entradas,
            dtype=np.float32
        )

        saidas = np.array(
            saidas,
            dtype=np.float32
        )

        entradas = entradas.reshape(
            entradas.shape[0],
            entradas.shape[1],
            1
        )

        saidas = saidas.reshape(-1, 1)

        return entradas, saidas

    def __dividir_treino_teste(
        self,
        entradas,
        saidas
    ):

        quantidade_treino = int(
            len(entradas) * self.__percentual_treino
        )

        x_treino = entradas[:quantidade_treino]

        y_treino = saidas[:quantidade_treino]

        x_teste = entradas[quantidade_treino:]

        y_teste = saidas[quantidade_treino:]

        return (
            x_treino,
            y_treino,
            x_teste,
            y_teste
        )

    def quantidade_amostras(self, serie):

        return len(serie) - self.__tamanho_janela

    def exibir_resumo(self, x_treino, x_teste):

        print()

        print("=" * 50)
        print("SEQUÊNCIAS GERADAS")
        print("=" * 50)

        print(f"Amostras de treino : {len(x_treino)}")
        print(f"Amostras de teste  : {len(x_teste)}")

        print(f"Janela temporal    : {self.__tamanho_janela}")

        print()