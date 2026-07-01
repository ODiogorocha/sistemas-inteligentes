from typing import Tuple 
from tensorflow.keras.datasets import cifar10

import numpy as np

class DatasetCifar10:

    def __init__(self) -> None:
        self._classes_originais = [0, 1, 8, 9]

        self._mapeamento_classes = {
            0: 0,
            1: 1,
            8: 2,
            9: 3
        }
        self._nomes_classes = [
            "Airplane",
            "Automobile",
            "Ship",
            "Truck"
        ]
    
    def carregar(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (imagens_treinameto,rotulos_treinamento
        ),(
        imagens_teste, rotulos_teste) = cifar10.load_data()

        imagens_treinameto, rotulos_treinamento = self._filtrar_classes(
            imagens_treinameto,
            rotulos_treinamento
        )

        imagens_teste, rotulos_teste = self._filtrar_classes(
            imagens_teste,
            rotulos_teste
        )

        imagens_treinameto = self._normalizar(imagens_treinameto)
        imagens_teste = self._normalizar(imagens_teste)

        return (imagens_treinameto,rotulos_treinamento,
                imagens_teste, rotulos_teste)
    
    def obter_nomes_classes(self) -> list[str]:
        return self._nomes_classes.copy()
    
    def quantidade_classes(self) -> int:
        return len(self._nomes_classes)
    
    def formato_entrada(self) -> tuple[int, int, int]:
        return (32, 32, 3)
    
    def _filtrar_classes(self, imagens: np.ndarray, rotulos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotulos = rotulos.flatten()
        mascara = np.isin(rotulos, self._classes_originais)

        imagens = imagens[mascara]
        rotulos = rotulos[mascara]

        rotulos = np.array([self._mapeamento_classes[classe]
                            for classe in rotulos])
        
        return imagens, rotulos
    
    @staticmethod
    def _normalizar(imagens: np.ndarray) -> np.ndarray:
        return imagens.astype("float32") / 255.0