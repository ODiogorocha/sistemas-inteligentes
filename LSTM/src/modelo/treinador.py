from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau


class Treinador:

    def __init__(
        self,
        epocas=100,
        tamanho_lote=16,
        percentual_validacao=0.2
    ):

        self.__epocas = epocas
        self.__tamanho_lote = tamanho_lote
        self.__percentual_validacao = percentual_validacao

    @property
    def epocas(self):

        return self.__epocas

    @property
    def tamanho_lote(self):

        return self.__tamanho_lote

    @property
    def percentual_validacao(self):

        return self.__percentual_validacao

    def treinar(
        self,
        modelo,
        x_treino,
        y_treino
    ):

        callbacks = self.__criar_callbacks()

        historico = modelo.fit(
            x_treino,
            y_treino,
            epochs=self.__epocas,
            batch_size=self.__tamanho_lote,
            validation_split=self.__percentual_validacao,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )

        return historico

    def __criar_callbacks(self):

        Path("resultados").mkdir(
            exist_ok=True
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduzir_taxa = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        )

        checkpoint = ModelCheckpoint(
            filepath="resultados/melhor_modelo.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        return [
            early_stopping,
            reduzir_taxa,
            checkpoint
        ]