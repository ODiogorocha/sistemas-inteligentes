from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential


class TreinadorCNN:

    def __init__(
        self,
        modelo: Sequential,
        diretorio_modelos: str = "resultados/modelos"
    ) -> None:

        self._modelo = modelo
        self._diretorio_modelos = Path(diretorio_modelos)
        self._diretorio_modelos.mkdir(
            parents=True,
            exist_ok=True
        )

    def treinar(
        self,
        imagens_treinamento: np.ndarray,
        rotulos_treinamento: np.ndarray,
        nome_modelo: str,
        epocas: int = 20,
        tamanho_lote: int = 64,
        percentual_validacao: float = 0.2
    ):

        callbacks = self._criar_callbacks(nome_modelo)

        historico = self._modelo.fit(
            x=imagens_treinamento,
            y=rotulos_treinamento,
            epochs=epocas,
            batch_size=tamanho_lote,
            validation_split=percentual_validacao,
            callbacks=callbacks,
            verbose=1
        )

        return historico

    def obter_modelo(self) -> Sequential:
        return self._modelo

    def salvar_modelo(
        self,
        nome_modelo: str
    ) -> None:

        caminho = self._diretorio_modelos / f"{nome_modelo}.keras"

        self._modelo.save(caminho)

    def _criar_callbacks(
        self,
        nome_modelo: str
    ) -> list:

        caminho = self._diretorio_modelos / f"{nome_modelo}.keras"

        return [

            ModelCheckpoint(
                filepath=caminho,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),

            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                verbose=1
            )

        ]