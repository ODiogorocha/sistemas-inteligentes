from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


class RedeLSTM:

    def __init__(
        self,
        quantidade_neuronios=64,
        taxa_dropout=0.2,
        taxa_aprendizado=0.001
    ):

        self.__quantidade_neuronios = quantidade_neuronios
        self.__taxa_dropout = taxa_dropout
        self.__taxa_aprendizado = taxa_aprendizado

    @property
    def quantidade_neuronios(self):

        return self.__quantidade_neuronios

    @property
    def taxa_dropout(self):

        return self.__taxa_dropout

    @property
    def taxa_aprendizado(self):

        return self.__taxa_aprendizado

    def criar(self, entrada):

        modelo = Sequential()

        modelo.add(
            LSTM(
                units=self.__quantidade_neuronios,
                input_shape=entrada,
                return_sequences=True
            )
        )

        modelo.add(
            Dropout(
                self.__taxa_dropout
            )
        )

        modelo.add(
            LSTM(
                units=self.__quantidade_neuronios // 2,
                return_sequences=False
            )
        )

        modelo.add(
            Dropout(
                self.__taxa_dropout
            )
        )

        modelo.add(
            Dense(
                units=32,
                activation="relu"
            )
        )

        modelo.add(
            Dense(
                units=16,
                activation="relu"
            )
        )

        modelo.add(
            Dense(
                units=1
            )
        )

        modelo.compile(
            optimizer=Adam(
                learning_rate=self.__taxa_aprendizado
            ),
            loss="mse",
            metrics=[
                "mae"
            ]
        )

        return modelo

    def exibir_resumo(self, modelo):

        print()

        print("=" * 50)
        print("ARQUITETURA DA REDE")
        print("=" * 50)

        modelo.summary()

        print()