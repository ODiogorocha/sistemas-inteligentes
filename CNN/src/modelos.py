from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


class ModelosCNN:

    def __init__(
        self,
        formato_entrada: tuple[int, int, int],
        quantidade_classes: int
    ) -> None:

        self._formato_entrada = formato_entrada
        self._quantidade_classes = quantidade_classes

    def criar_modelo_1(self) -> Sequential:

        modelo = Sequential([
            Input(shape=self._formato_entrada),

            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu"
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu"
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(
                units=128,
                activation="relu"
            ),

            Dense(
                units=self._quantidade_classes,
                activation="softmax"
            )
        ])

        return self._compilar(modelo)

    def criar_modelo_2(self) -> Sequential:

        modelo = Sequential([
            Input(shape=self._formato_entrada),

            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu"
            ),

            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation="relu"
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu"
            ),

            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                activation="relu"
            ),

            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(
                units=256,
                activation="relu"
            ),

            Dropout(0.4),

            Dense(
                units=self._quantidade_classes,
                activation="softmax"
            )
        ])

        return self._compilar(modelo)

    def criar_modelo_3(self) -> Sequential:

        modelo = Sequential([
        Input(shape=self._formato_entrada),

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        ),

        Flatten(),

        Dense(
            units=512,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            units=256,
            activation="relu"
        ),

        Dense(
            units=self._quantidade_classes,
            activation="softmax"
        )
    ])

        return self._compilar(modelo)

    @staticmethod
    def _compilar(modelo: Sequential) -> Sequential:

        modelo.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return modelo