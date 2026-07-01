import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class Avaliador:

    def __init__(self):

        self.__resultado = None

    @property
    def resultado(self):

        return self.__resultado

    def avaliar(
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

        mse = mean_squared_error(
            valores_reais,
            valores_previstos
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            valores_reais,
            valores_previstos
        )

        self.__resultado = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae)
        }

        return self.__resultado

    def exibir(self, resultado):

        print()

        print("=" * 50)
        print("RESULTADOS DA AVALIAÇÃO")
        print("=" * 50)

        print(f"MSE  : {resultado['mse']:.4f}")
        print(f"RMSE : {resultado['rmse']:.4f}")
        print(f"MAE  : {resultado['mae']:.4f}")

        print()

    def obter_mse(self):

        self.__verificar_resultado()

        return self.__resultado["mse"]

    def obter_rmse(self):

        self.__verificar_resultado()

        return self.__resultado["rmse"]

    def obter_mae(self):

        self.__verificar_resultado()

        return self.__resultado["mae"]

    def __verificar_resultado(self):

        if self.__resultado is None:

            raise RuntimeError(
                "Nenhuma avaliação foi realizada."
            )