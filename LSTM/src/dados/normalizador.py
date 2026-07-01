import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Normalizador:

    def __init__(self):

        self.__scaler = MinMaxScaler(feature_range=(0, 1))

        self.__ajustado = False

    @property
    def scaler(self):

        return self.__scaler

    def normalizar(self, serie):

        serie = self.__converter_para_coluna(serie)

        serie_normalizada = self.__scaler.fit_transform(serie)

        self.__ajustado = True

        return serie_normalizada

    def transformar(self, serie):

        self.__verificar_ajuste()

        serie = self.__converter_para_coluna(serie)

        return self.__scaler.transform(serie)

    def desnormalizar(self, serie):

        self.__verificar_ajuste()

        serie = self.__converter_para_coluna(serie)

        return self.__scaler.inverse_transform(serie)

    def __converter_para_coluna(self, serie):

        serie = np.asarray(serie, dtype=np.float32)

        return serie.reshape(-1, 1)

    def __verificar_ajuste(self):

        if not self.__ajustado:

            raise RuntimeError(
                "O normalizador ainda não foi ajustado."
            )