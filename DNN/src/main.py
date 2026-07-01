import pandas as pd
import numpy as np
import sys
import traceback
from pathlib import Path

from preprocessamento import Preprocessamento
from selecao_features import SelecaoFeatures
from classificacao_binaria import ClassificacaoBinaria
from regressao import Regressao
from metricas import Metricas
from metricas_regressao import MetricasRegressao
from graficos_regressao import GraficosRegressao
from visualizacao import Visualizacao
from optuna_otimizacao import OtimizacaoOptuna
from regularizacao import executar_analise_regularizacao


class PipelineMachineLearning:
    
    def __init__(self, caminho_dados, diretorio_resultados):
        self.caminho_dados = Path(caminho_dados)
        self.diretorio_resultados = Path(diretorio_resultados)
        self.dados = None
        self.features = None
        self.coluna_alvo_classificacao = 'target'
        self.coluna_alvo_regressao = 'age'
        
        self._criar_diretorios()
    
    def _criar_diretorios(self):
        subdiretorios = [
            'classificacao_binaria',
            'regressao',
            'optuna',
            'selecao_features',
            'regularizacao'
        ]
        for subdir in subdiretorios:
            (self.diretorio_resultados / subdir).mkdir(parents=True, exist_ok=True)
    
    def carregar_dados(self):
        try:
            self.dados = pd.read_csv(self.caminho_dados)
            print(f" Dataset carregado com sucesso!")
            print(f"   Amostras: {self.dados.shape[0]}")
            print(f"   Atributos: {self.dados.shape[1]}")
            print(f"   Colunas: {', '.join(self.dados.columns)}")
            return self.dados
        except Exception as erro:
            print(f" Erro ao carregar dados: {erro}")
            sys.exit(1)
    
    def _salvar_informacoes_dataset(self, features, X_treino, X_teste):

        with open(self.diretorio_resultados / 'dataset_info.txt', 'w') as arquivo:
            arquivo.write("=" * 60 + "\n")
            arquivo.write("INFORMAÇÕES DO DATASET\n")
            arquivo.write("=" * 60 + "\n\n")
            arquivo.write(f"Nome do dataset: {self.caminho_dados.name}\n")
            arquivo.write(f"Total de amostras: {len(self.dados)}\n")
            arquivo.write(f"Total de atributos: {len(features)}\n")
            arquivo.write(f"Atributos: {', '.join(features)}\n")
            arquivo.write(f"Alvo (classificação): {self.coluna_alvo_classificacao}\n")
            arquivo.write(f"Alvo (regressão): {self.coluna_alvo_regressao}\n")
            arquivo.write(f"Amostras treino: {X_treino.shape[0]}\n")
            arquivo.write(f"Amostras teste: {X_teste.shape[0]}\n")
            arquivo.write("\nDistribuição das classes:\n")
            for classe, quantidade in self.dados[self.coluna_alvo_classificacao].value_counts().items():
                arquivo.write(f"  Classe {classe}: {quantidade} ({quantidade/len(self.dados)*100:.1f}%)\n")
    
    def executar_preprocessamento(self):

        print("\n" + "=" * 60)
        print("ETAPA 1: PRÉ-PROCESSAMENTO")
        print("=" * 60)
        
        X_treino, X_teste, y_treino_class, y_teste_class, escalador, features = (Preprocessamento.executar(self.dados,alvo=self.coluna_alvo_classificacao))
        
        indices_treino = X_treino.index if hasattr(X_treino, 'index') else range(len(X_treino))
        indices_teste = X_teste.index if hasattr(X_teste, 'index') else range(len(X_teste))
        
        y_treino_reg = self.dados.iloc[indices_treino][self.coluna_alvo_regressao].values
        y_teste_reg = self.dados.iloc[indices_teste][self.coluna_alvo_regressao].values
        
        if not isinstance(X_treino, np.ndarray):
            X_treino = X_treino.values if hasattr(X_treino, 'values') else np.array(X_treino)
        if not isinstance(X_teste, np.ndarray):
            X_teste = X_teste.values if hasattr(X_teste, 'values') else np.array(X_teste)
        
        self.features = features
        
        self._salvar_informacoes_dataset(features, X_treino, X_teste)
        
        return X_treino, X_teste, y_treino_class, y_teste_class, y_treino_reg, y_teste_reg, escalador, features
    
    def executar_selecao_features(self, X_treino, X_teste, y_treino, features):
        
        print("\n" + "=" * 60)
        print("ETAPA 2: SELEÇÃO DE FEATURES")
        print("=" * 60)
        
        seletor = SelecaoFeatures()
        
        X_treino_selecionado, X_teste_selecionado, features_selecionadas = (
            seletor.selecionar_por_importancia_random_forest(X_treino,X_teste,y_treino,features,quantidade_features=10))
        
        seletor.plotar_importancia(diretorio=self.diretorio_resultados / 'selecao_features')
        
        seletor.salvar_features_selecionadas(self.diretorio_resultados / 'selecao_features' / 'features_selecionadas.txt')
        
        print(f"\n {len(features_selecionadas)} features selecionadas:")
        for i, feature in enumerate(features_selecionadas, 1):
            print(f"   {i}. {feature}")
        
        return X_treino_selecionado, X_teste_selecionado, features_selecionadas
    
    def executar_classificacao(self, X_treino, X_teste, y_treino, y_teste, features, nome_experimento):

        print("\n" + "=" * 60)
        print(f"ETAPA 3: CLASSIFICAÇÃO - {nome_experimento}")
        print("=" * 60)
        
        configuracao = {
            'camadas_ocultas': [64, 32, 16],
            'ativacao': 'relu',
            'otimizador': 'adam',
            'taxa_aprendizado': 0.001,
            'epocas': 100,
            'tamanho_batch': 32,
            'dropout': 0.3
        }
        
        print("\n Treinando modelo de classificação...")
        modelo, historico = ClassificacaoBinaria.treinar(X_treino, y_treino,X_teste, y_teste,**configuracao)
        
        y_pred_proba = modelo.predict(X_teste)
        y_pred_classes = (y_pred_proba > 0.5).astype(int)
        
        metricas = Metricas.calcular_todas(y_teste, y_pred_classes, y_pred_proba)
        
        caminho_metricas = (self.diretorio_resultados / 'classificacao_binaria' / f'metricas_{nome_experimento.replace(" ", "_")}.txt')
        Metricas.salvar_metricas(metricas, caminho_metricas)
        
        diretorio_graficos = self.diretorio_resultados / 'classificacao_binaria'
        Visualizacao.plotar_curva_aprendizado(historico, diretorio_graficos, nome_experimento)
        Visualizacao.plotar_matriz_confusao(y_teste, y_pred_classes, diretorio_graficos, nome_experimento)
        Visualizacao.plotar_curva_roc(y_teste, y_pred_proba, diretorio_graficos, nome_experimento)
        
        print("\n Métricas de Classificação:")
        for chave, valor in metricas.items():
            if chave != 'confusion_matrix':
                print(f"   {chave}: {valor:.4f}")
        
        return modelo, historico
    
    def executar_regressao(self, X_treino, X_teste, y_treino, y_teste, features):

        print("\n" + "=" * 60)
        print("ETAPA 4: REGRESSÃO")
        print("=" * 60)
        
        configuracao = {
            'camadas_ocultas': [64, 32, 16],
            'ativacao': 'relu',
            'otimizador': 'adam',
            'taxa_aprendizado': 0.001,
            'epocas': 100,
            'tamanho_batch': 32
        }
        
        print("\n Treinando modelo de regressão...")
        modelo, historico = Regressao.treinar(X_treino, y_treino,X_teste, y_teste,**configuracao)
        y_pred = modelo.predict(X_teste).flatten()
        metricas = MetricasRegressao.calcular_todas(y_teste, y_pred)
        
        caminho_metricas = self.diretorio_resultados / 'regressao' / 'metricas.txt'
        MetricasRegressao.salvar_metricas(metricas, caminho_metricas)
        
        diretorio_graficos = self.diretorio_resultados / 'regressao'
        GraficosRegressao.plotar_valores_reais_vs_preditos(y_teste, y_pred, diretorio_graficos)
        GraficosRegressao.plotar_residuos(y_teste, y_pred, diretorio_graficos)
        GraficosRegressao.plotar_curva_aprendizado(historico, diretorio_graficos)
        
        print("\n Métricas de Regressão:")
        for chave, valor in metricas.items():
            print(f"   {chave}: {valor:.4f}")
        
        return modelo, historico
    
    def executar_otimizacao_optuna(self, X_treino, X_teste, y_treino, y_teste):

        print("\n" + "=" * 60)
        print("ETAPA 5: OTIMIZAÇÃO COM OPTUNA")
        print("=" * 60)
        
        otimizador = OtimizacaoOptuna()
        
        print("\n Executando otimização (50 testes)...")
        melhores_params = otimizador.otimizar(X_treino, y_treino,X_teste, y_teste,numero_testes=50)
        otimizador.salvar_resultados(self.diretorio_resultados / 'optuna' / 'melhores_hiperparametros.json')
        
        print("\n Melhores hiperparâmetros encontrados:")
        for chave, valor in melhores_params.items():
            print(f"   {chave}: {valor}")
        
        return melhores_params
    
    def executar_regularizacao(self, X_treino, X_teste, y_treino, y_teste):

        print("\n" + "=" * 60)
        print("ETAPA 6: REGULARIZAÇÃO E OVERFITTING")
        print("=" * 60)
        
        from sklearn.model_selection import train_test_split
        X_treino_reg, X_validacao_reg, y_treino_reg, y_validacao_reg = train_test_split(
            X_treino, y_treino,
            test_size=0.2,
            random_state=42,
            stratify=y_treino
        )
        
        analise = executar_analise_regularizacao(
            self.diretorio_resultados,
            X_treino_reg, X_validacao_reg, X_teste,
            y_treino_reg, y_validacao_reg, y_teste
        )
        
        return analise
    
    def executar(self):
        try:
            self.carregar_dados()
            
            (X_treino, X_teste, y_treino_class, y_teste_class,
            y_treino_reg, y_teste_reg, _, features) = self.executar_preprocessamento()
            
            X_treino_selecionado, X_teste_selecionado, features_selecionadas = (self.executar_selecao_features(X_treino, X_teste, y_treino_class, features))
            
            _, _ = self.executar_classificacao(
                X_treino, X_teste,
                y_treino_class, y_teste_class,
                features,
                "Todas Features"
            )
            
            _, _ = self.executar_classificacao(
                X_treino_selecionado, X_teste_selecionado,
                y_treino_class, y_teste_class,
                features_selecionadas,
                "Features Selecionadas"
            )
            
            _, _ = self.executar_regressao(
                X_treino, X_teste,
                y_treino_reg, y_teste_reg,
                features
            )
            
            _ = self.executar_otimizacao_optuna(
                X_treino, X_teste,
                y_treino_class, y_teste_class
            )
            
            _ = self.executar_regularizacao(
                X_treino, X_teste,
                y_treino_class, y_teste_class
            )
            
            print("\n" + "=" * 60)
            print(" PIPELINE EXECUTADO COM SUCESSO!")
            print("=" * 60)
            print(f" Resultados salvos em: {self.diretorio_resultados}")
            
        except Exception as erro:
            print(f"\n Erro durante a execução do pipeline: {erro}")
            traceback.print_exc()
            sys.exit(1)


def main():

    diretorio_base = Path(__file__).parent.parent
    caminho_dados = diretorio_base / 'dados' / 'heart.csv'
    diretorio_resultados = diretorio_base / 'resultados'
    
    pipeline = PipelineMachineLearning(caminho_dados, diretorio_resultados)
    pipeline.executar()


if __name__ == "__main__":
    main()