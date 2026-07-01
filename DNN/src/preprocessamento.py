import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class Preprocessamento:

    @staticmethod
    def executar(dados, alvo, tamanho_teste=0.2, semente_aleatoria=42):
        if alvo not in dados.columns:
            raise ValueError(f"Coluna alvo '{alvo}' não encontrada no dataset")
        
        X = dados.drop(columns=[alvo])
        y = dados[alvo]
        nomes_features = X.columns.tolist()
        
        X = Preprocessamento._codificar_categoricas(X)
        X_treino, X_teste, y_treino, y_teste = Preprocessamento._dividir_dados(X, y, tamanho_teste, semente_aleatoria)
        X_treino_escalado, X_teste_escalado, escalador = Preprocessamento._padronizar_dados(X_treino, X_teste)
        
        return X_treino_escalado, X_teste_escalado, y_treino, y_teste, escalador, nomes_features
    
    @staticmethod
    def _codificar_categoricas(X):
        colunas_categoricas = X.select_dtypes(include=['object']).columns.tolist()
        
        if colunas_categoricas:
            print(f"📝 Codificando variáveis categóricas: {', '.join(colunas_categoricas)}")
            
            for coluna in colunas_categoricas:
                codificador = LabelEncoder()
                X[coluna] = codificador.fit_transform(X[coluna].astype(str))
        
        return X
    
    @staticmethod
    def _dividir_dados(X, y, tamanho_teste, semente_aleatoria):

        estratificar = None
        if len(y.unique()) <= 10: 
            estratificar = y
        
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y,
            test_size=tamanho_teste,
            random_state=semente_aleatoria,
            stratify=estratificar
        )
        
        print(f" Divisão dos dados:")
        print(f"   Treino: {X_treino.shape[0]} amostras")
        print(f"   Teste: {X_teste.shape[0]} amostras")
        
        return X_treino, X_teste, y_treino, y_teste
    
    @staticmethod
    def _padronizar_dados(X_treino, X_teste):
        escalador = StandardScaler()
        X_treino_escalado = escalador.fit_transform(X_treino)
        X_teste_escalado = escalador.transform(X_teste)
        
        print(f" Dados padronizados (média=0, desvio=1)")
        
        return X_treino_escalado, X_teste_escalado, escalador
    
    @staticmethod
    def tratar_valores_ausentes(dados, estrategia='media'):

        if dados.isnull().sum().sum() == 0:
            print(" Nenhum valor ausente encontrado")
            return dados
        
        print(f" Valores ausentes encontrados: {dados.isnull().sum().sum()}")
        
        if estrategia == 'remover':
            dados = dados.dropna()
            print(f"   Registros removidos: {dados.shape[0]}")
        else:
            for coluna in dados.columns:
                if dados[coluna].isnull().any():
                    if estrategia == 'media':
                        valor_preenchimento = dados[coluna].mean()
                    elif estrategia == 'mediana':
                        valor_preenchimento = dados[coluna].median()
                    elif estrategia == 'moda':
                        valor_preenchimento = dados[coluna].mode()[0]
                    else:
                        raise ValueError(f"Estratégia '{estrategia}' não suportada")
                    
                    dados[coluna].fillna(valor_preenchimento, inplace=True)
                    print(f"   Coluna '{coluna}': preenchida com {estrategia} = {valor_preenchimento:.2f}")
        
        return dados