import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class Regressao:
    @staticmethod
    def criar_modelo(entrada_shape, camadas_ocultas, ativacao):

        modelo = keras.Sequential()
        modelo.add(layers.Input(shape=(entrada_shape,)))
        
        for neuronios in camadas_ocultas:
            modelo.add(layers.Dense(neuronios, activation=ativacao))
        
        modelo.add(layers.Dense(1))
        
        return modelo
    
    @staticmethod
    def treinar(X_treino, y_treino, X_validacao, y_validacao,
                camadas_ocultas=[64, 32, 16],
                ativacao='relu',
                otimizador='adam',
                taxa_aprendizado=0.001,
                epocas=100,
                tamanho_batch=32,
                early_stopping=True):

        modelo = Regressao.criar_modelo(X_treino.shape[1],camadas_ocultas,ativacao)
        
        if otimizador == 'adam':
            otimizador_obj = keras.optimizers.Adam(learning_rate=taxa_aprendizado)
        elif otimizador == 'sgd':
            otimizador_obj = keras.optimizers.SGD(learning_rate=taxa_aprendizado)
        elif otimizador == 'rmsprop':
            otimizador_obj = keras.optimizers.RMSprop(learning_rate=taxa_aprendizado)
        else:
            raise ValueError(f"Otimizador '{otimizador}' não suportado")
        
        modelo.compile(
            optimizer=otimizador_obj,
            loss='mse',
            metrics=['mae']
        )
        
        callbacks = []
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        print(f"\n Iniciando treinamento de regressão...")
        print(f"   Arquitetura: {len(camadas_ocultas)} camadas ocultas")
        print(f"   Neurônios: {camadas_ocultas}")
        print(f"   Ativação: {ativacao}")
        print(f"   Otimizador: {otimizador} (lr={taxa_aprendizado})")
        print(f"   Épocas: {epocas}, Batch: {tamanho_batch}")
        
        historico = modelo.fit(
            X_treino, y_treino,
            validation_data=(X_validacao, y_validacao),
            epochs=epocas,
            batch_size=tamanho_batch,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f" Treinamento de regressão concluído!")
        
        return modelo, historico
    
    @staticmethod
    def avaliar(modelo, X_teste, y_teste):
        mse, mae = modelo.evaluate(X_teste, y_teste, verbose=0)
        return mse, mae