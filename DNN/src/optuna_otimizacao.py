import optuna
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class OtimizacaoOptuna:    
    def __init__(self):
        self.melhores_params = None
        self.melhor_score = None
        self.estudo = None
    
    def _criar_modelo(self, trial, entrada_shape):
        
        num_camadas = trial.suggest_int('num_camadas', 1, 4)
        num_neuronios = trial.suggest_int('num_neuronios', 16, 128, step=16)
        taxa_aprendizado = trial.suggest_float('taxa_aprendizado', 1e-5, 1e-2, log=True)
        taxa_dropout = trial.suggest_float('taxa_dropout', 0.0, 0.5)
        ativacao = trial.suggest_categorical('ativacao', ['relu', 'tanh', 'sigmoid'])
        otimizador = trial.suggest_categorical('otimizador', ['adam', 'sgd', 'rmsprop'])
        
        modelo = keras.Sequential()
        modelo.add(layers.Input(shape=(entrada_shape,)))
        
        modelo.add(layers.Dense(num_neuronios, activation=ativacao))
        if taxa_dropout > 0:
            modelo.add(layers.Dropout(taxa_dropout))
        
        for i in range(1, num_camadas):
            neuronios = trial.suggest_int(f'neuronios_camada_{i}', 16, num_neuronios, step=16)
            modelo.add(layers.Dense(neuronios, activation=ativacao))
            if taxa_dropout > 0:
                modelo.add(layers.Dropout(taxa_dropout))
        
        modelo.add(layers.Dense(1, activation='sigmoid'))
        
        if otimizador == 'adam':
            otimizador_obj = keras.optimizers.Adam(learning_rate=taxa_aprendizado)
        elif otimizador == 'sgd':
            otimizador_obj = keras.optimizers.SGD(learning_rate=taxa_aprendizado)
        elif otimizador == 'rmsprop':
            otimizador_obj = keras.optimizers.RMSprop(learning_rate=taxa_aprendizado)
        
        modelo.compile(
            optimizer=otimizador_obj,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return modelo
    
    def _objetivo(self, trial, X_treino, y_treino, X_validacao, y_validacao):
        modelo = self._criar_modelo(trial, X_treino.shape[1])
        
        tamanho_batch = trial.suggest_categorical('tamanho_batch', [16, 32, 64])
        epocas = trial.suggest_int('epocas', 50, 200, step=50)
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        
        historico = modelo.fit(
            X_treino, y_treino,
            validation_data=(X_validacao, y_validacao),
            batch_size=tamanho_batch,
            epochs=epocas,
            callbacks=[early_stop],
            verbose=0
        )
        
        _, acuracia = modelo.evaluate(X_validacao, y_validacao, verbose=0)
        
        return acuracia
    
    def otimizar(self, X_treino, y_treino, X_validacao, y_validacao, numero_testes=50):
        self.estudo = optuna.create_study(
            direction='maximize',
            study_name='otimizacao_mlp',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.estudo.optimize(
            lambda trial: self._objetivo(trial, X_treino, y_treino, X_validacao, y_validacao),n_trials=numero_testes)
        
        self.melhores_params = self.estudo.best_params
        self.melhor_score = self.estudo.best_value
        
        print(f"\n Otimização concluída!")
        print(f"   Melhor acurácia: {self.melhor_score:.4f}")
        print(f"   Número de testes: {len(self.estudo.trials)}")
        
        self._exibir_analise()
        
        return self.melhores_params
    
    def _exibir_analise(self):
        print("\n Análise dos hiperparâmetros:")
        
        importancias = optuna.importance.get_param_importances(self.estudo)
        
        print("   Importância dos hiperparâmetros:")
        for param, importancia in sorted(importancias.items(), key=lambda x: x[1], reverse=True):
            print(f"      {param}: {importancia:.3f}")
    
    def salvar_resultados(self, caminho_arquivo):
        if self.melhores_params is None:
            raise ValueError("Nenhum resultado para salvar. Execute otimização primeiro.")
        
        resultados = {
            'melhores_hiperparametros': self.melhores_params,
            'melhor_score': float(self.melhor_score),
            'num_testes': len(self.estudo.trials) if self.estudo else 0,
            'importancia_hiperparametros': {
                param: float(importancia)
                for param, importancia in optuna.importance.get_param_importances(self.estudo).items()
            }
        }
        
        with open(caminho_arquivo, 'w') as arquivo:
            json.dump(resultados, arquivo, indent=4)
        
        print(f" Resultados salvos em: {caminho_arquivo}")