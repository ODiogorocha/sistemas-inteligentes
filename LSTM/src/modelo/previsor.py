import numpy as np


class Previsor:

    def __init__(self):

        self.__previsoes = None

    @property
    def previsoes(self):

        return self.__previsoes

    def prever(
        self,
        modelo,
        x_teste
    ):

        self.__previsoes = modelo.predict(
            x_teste,
            verbose=0
        )

        return self.__previsoes

    def prever_proximo_valor(
        self,
        modelo,
        ultima_sequencia
    ):

        entrada = np.asarray(
            ultima_sequencia,
            dtype=np.float32
        )

        entrada = entrada.reshape(
            1,
            entrada.shape[0],
            1
        )

        previsao = modelo.predict(
            entrada,
            verbose=0
        )

        return float(previsao[0][0])

    def prever_varios_passos(
        self,
        modelo,
        ultima_sequencia,
        quantidade_passos
    ):

        sequencia = list(
            np.asarray(
                ultima_sequencia,
                dtype=np.float32
            ).flatten()
        )

        previsoes = []

        for _ in range(quantidade_passos):

            entrada = np.array(
                sequencia,
                dtype=np.float32
            ).reshape(
                1,
                len(sequencia),
                1
            )

            proximo_valor = modelo.predict(
                entrada,
                verbose=0
            )[0][0]

            previsoes.append(proximo_valor)

            sequencia.pop(0)

            sequencia.append(proximo_valor)

        return np.array(
            previsoes,
            dtype=np.float32
        ).reshape(-1, 1)

    def exibir_previsoes(
        self,
        valores_reais,
        valores_previstos,
        quantidade=10
    ):

        print()

        print("=" * 60)
        print("COMPARAÇÃO ENTRE VALORES REAIS E PREVISTOS")
        print("=" * 60)

        limite = min(
            quantidade,
            len(valores_reais)
        )

        for indice in range(limite):

            real = float(valores_reais[indice])

            previsto = float(valores_previstos[indice])

            print(
                f"{indice + 1:02d} | "
                f"Real: {real:.2f} | "
                f"Previsto: {previsto:.2f}"
            )

        print()