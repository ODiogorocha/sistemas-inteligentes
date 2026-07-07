import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from pathlib import Path


class AnaliseRegularizacao:
    def __init__(self, diretorio_resultados):
        self.diretorio = Path(diretorio_resultados)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        self.resultados = {}
        self.modelos = {}
        self.historicos = {}
        
    def _criar_modelo_base(self, entrada_shape, saida_shape=1):
        modelo = keras.Sequential([
            layers.Input(shape=(entrada_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(saida_shape, activation='sigmoid')
        ])
        return modelo
    
    def _criar_modelo_com_dropout(self, entrada_shape, saida_shape=1, taxa_dropout=0.3):
        modelo = keras.Sequential([
            layers.Input(shape=(entrada_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(taxa_dropout),
            layers.Dense(32, activation='relu'),
            layers.Dropout(taxa_dropout),
            layers.Dense(16, activation='relu'),
            layers.Dropout(taxa_dropout),
            layers.Dense(saida_shape, activation='sigmoid')
        ])
        return modelo
    
    def _criar_modelo_com_l2(self, entrada_shape, saida_shape=1, peso_l2=0.001):
        modelo = keras.Sequential([
            layers.Input(shape=(entrada_shape,)),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(peso_l2)),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(peso_l2)),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(peso_l2)),
            layers.Dense(saida_shape, activation='sigmoid')
        ])
        return modelo
    
    def _criar_modelo_simples(self, entrada_shape, saida_shape=1):

        modelo = keras.Sequential([
            layers.Input(shape=(entrada_shape,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(saida_shape, activation='sigmoid')
        ])
        return modelo
    
    def _criar_modelo_com_dropout_l2(self, entrada_shape, saida_shape=1, taxa_dropout=0.3, peso_l2=0.001):

        modelo = keras.Sequential([
            layers.Input(shape=(entrada_shape,)),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(peso_l2)),
            layers.Dropout(taxa_dropout),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(peso_l2)),
            layers.Dropout(taxa_dropout),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(peso_l2)),
            layers.Dropout(taxa_dropout),
            layers.Dense(saida_shape, activation='sigmoid')
        ])
        return modelo
    
    def _treinar_modelo(self, modelo, X_treino, y_treino, X_validacao, y_validacao,nome, epocas=150, tamanho_batch=32):

        modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        print(f"\n Treinando modelo: {nome}")
        historico = modelo.fit(
            X_treino, y_treino,
            validation_data=(X_validacao, y_validacao),
            epochs=epocas,
            batch_size=tamanho_batch,
            callbacks=[early_stop],
            verbose=1
        )
        
        return historico
    
    def executar_analise(self, X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste):

        print("\n" + "=" * 60)
        print("ETAPA 7: REGULARIZAÇÃO E ANÁLISE DE OVERFITTING")
        print("=" * 60)
        
        entrada_shape = X_treino.shape[1]
        
        self.resultados = {
            'modelos': {},
            'metricas': {},
            'historico': {}
        }
        
        modelo_base = self._criar_modelo_base(entrada_shape)
        historico_base = self._treinar_modelo(
            modelo_base, X_treino, y_treino, X_validacao, y_validacao,
            "Modelo Base (Sem Regularização)"
        )
        self.modelos['base'] = modelo_base
        self.historicos['base'] = historico_base
        
        modelo_dropout = self._criar_modelo_com_dropout(entrada_shape, taxa_dropout=0.3)
        historico_dropout = self._treinar_modelo(
            modelo_dropout, X_treino, y_treino, X_validacao, y_validacao,
            "Modelo com Dropout (0.3)"
        )
        self.modelos['dropout'] = modelo_dropout
        self.historicos['dropout'] = historico_dropout
        
        modelo_l2 = self._criar_modelo_com_l2(entrada_shape, peso_l2=0.001)
        historico_l2 = self._treinar_modelo(
            modelo_l2, X_treino, y_treino, X_validacao, y_validacao,
            "Modelo com L2 (0.001)"
        )
        self.modelos['l2'] = modelo_l2
        self.historicos['l2'] = historico_l2
        
        modelo_simples = self._criar_modelo_simples(entrada_shape)
        historico_simples = self._treinar_modelo(
            modelo_simples, X_treino, y_treino, X_validacao, y_validacao,
            "Modelo Simplificado"
        )
        self.modelos['simples'] = modelo_simples
        self.historicos['simples'] = historico_simples
        
        modelo_combo = self._criar_modelo_com_dropout_l2(entrada_shape, taxa_dropout=0.3, peso_l2=0.001)
        historico_combo = self._treinar_modelo(modelo_combo, X_treino, y_treino, X_validacao, y_validacao,"Modelo com Dropout + L2")
        self.modelos['combo'] = modelo_combo
        self.historicos['combo'] = historico_combo
        
        self._avaliar_modelos(X_teste, y_teste)
        self._plotar_curvas_aprendizado()
        self._plotar_comparacao_metricas()
        self._salvar_relatorio()
        self._exibir_analise()
    
    def _avaliar_modelos(self, X_teste, y_teste):

        print("\n Avaliando modelos no conjunto de teste...")
        
        for nome, modelo in self.modelos.items():
            y_pred_proba = modelo.predict(X_teste)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            acuracia = accuracy_score(y_teste, y_pred)
            precisao = precision_score(y_teste, y_pred)
            recall = recall_score(y_teste, y_pred)
            f1 = f1_score(y_teste, y_pred)
            
            hist = self.historicos[nome]
            acc_treino = hist.history['accuracy'][-1]
            acc_val = hist.history['val_accuracy'][-1]
            diff_overfit = acc_treino - acc_val
            
            self.resultados['metricas'][nome] = {
                'acuracia': acuracia,
                'precisao': precisao,
                'recall': recall,
                'f1': f1,
                'acc_treino': acc_treino,
                'acc_val': acc_val,
                'diff_overfit': diff_overfit
            }
            
            print(f"\n   {nome}:")
            print(f"      Acurácia Teste: {acuracia:.4f}")
            print(f"      Acurácia Treino: {acc_treino:.4f}")
            print(f"      Acurácia Validação: {acc_val:.4f}")
            print(f"      Diferença (Treino - Val): {diff_overfit:.4f}")
    
    def _plotar_curvas_aprendizado(self):
        """
        """
        print("\nGerando curvas de aprendizado...")
    
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Cores para treino e validação
        cores = {'treino': '#2E86AB', 'validacao': '#A23B72'}
        
        for idx, (nome, historico) in enumerate(self.historicos.items()):
            if idx >= 6:  # Limitar a 6 gráficos
                break
                
            ax = axes[idx]
            
            ax.plot(historico.history['loss'], label='Loss Treino', color=cores['treino'], linewidth=2)
            ax.plot(historico.history['val_loss'], label='Loss Validação', color=cores['validacao'], linewidth=2, linestyle='--')
            
            ax2 = ax.twinx()
            ax2.plot(historico.history['accuracy'], label='Acc Treino', color=cores['treino'], linewidth=2, alpha=0.5)
            ax2.plot(historico.history['val_accuracy'], label='Acc Validação', color=cores['validacao'], linewidth=2, alpha=0.5, linestyle='--')
            
            ax.set_xlabel('Época', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10, color='black')
            ax2.set_ylabel('Acurácia', fontsize=10, color='black')
            
            titulo = nome.replace('Modelo ', '').replace('com ', '')
            ax.set_title(f'{titulo}', fontsize=12, fontweight='bold')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            
            ax.grid(alpha=0.3)
        
        for idx in range(len(self.historicos), 6):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Curvas de Aprendizado - Comparação de Regularizações', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.diretorio / 'curvas_aprendizado_comparacao.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico salvo: {self.diretorio / 'curvas_aprendizado_comparacao.png'}")
    
    def _plotar_comparacao_metricas(self):

        print("\nGerando gráfico de comparação de métricas...")

        modelos = list(self.resultados['metricas'].keys())
        metricas = ['acuracia', 'precisao', 'recall', 'f1']
        
        dados = []
        for modelo in modelos:
            for metrica in metricas:
                dados.append({
                    'Modelo': modelo,
                    'Métrica': metrica.capitalize(),
                    'Valor': self.resultados['metricas'][modelo][metrica]
                })
        
        df = pd.DataFrame(dados)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(modelos))
        width = 0.2
        multiplier = 0
        
        cores = ['#2E86AB', '#A23B72', '#F18F01', '#048A81']
        
        for metrica, cor in zip(metricas, cores):
            offset = width * multiplier
            valores = [self.resultados['metricas'][m][metrica] for m in modelos]
            ax.bar(x + offset, valores, width, label=metrica.capitalize(), color=cor)
            multiplier += 1
        
        # Configurar gráfico
        ax.set_ylabel('Valor', fontsize=12)
        ax.set_title('Comparação de Métricas entre Modelos', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([m.replace('Modelo ', '').replace('com ', '') for m in modelos], rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.diretorio / 'comparacao_metricas.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico salvo: {self.diretorio / 'comparacao_metricas.png'}")
    
    def _plotar_diferenca_overfit(self):

        print("\n Gerando gráfico de análise de overfitting...")
        
        modelos = list(self.resultados['metricas'].keys())
        diffs = [self.resultados['metricas'][m]['diff_overfit'] for m in modelos]
        acc_teste = [self.resultados['metricas'][m]['acuracia'] for m in modelos]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(modelos))
        cores = ['#2E86AB' if d < 0.1 else '#A23B72' for d in diffs]
        
        ax.bar(x, diffs, color=cores, alpha=0.7, label='Diferença Treino-Val')
        
        ax2 = ax.twinx()
        ax2.plot(x, acc_teste, 'o-', color='#048A81', linewidth=2, markersize=8, label='Acurácia Teste')
        
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel('Diferença Treino - Validação', fontsize=12)
        ax2.set_ylabel('Acurácia no Teste', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('Modelo ', '').replace('com ', '') for m in modelos], rotation=45, ha='right')
        
        # Adicionar linha de referência
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Limite Overfitting')
        
        # Legendas
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(alpha=0.3)
        ax.set_title('Análise de Overfitting por Modelo', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.diretorio / 'analise_overfitting.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico salvo: {self.diretorio / 'analise_overfitting.png'}")
    
    def _salvar_relatorio(self):

        print("\n Gerando relatório...")
        
        with open(self.diretorio / 'relatorio_regularizacao.txt', 'w') as arquivo:
            arquivo.write("=" * 80 + "\n")
            arquivo.write("RELATÓRIO DE REGULARIZAÇÃO E ANÁLISE DE OVERFITTING\n")
            arquivo.write("=" * 80 + "\n\n")
            
            arquivo.write("1. RESULTADOS DOS MODELOS\n")
            arquivo.write("-" * 60 + "\n\n")
            
            for nome, metricas in self.resultados['metricas'].items():
                arquivo.write(f"Modelo: {nome}\n")
                arquivo.write(f"  Acurácia Treino: {metricas['acc_treino']:.4f}\n")
                arquivo.write(f"  Acurácia Validação: {metricas['acc_val']:.4f}\n")
                arquivo.write(f"  Acurácia Teste: {metricas['acuracia']:.4f}\n")
                arquivo.write(f"  Precisão: {metricas['precisao']:.4f}\n")
                arquivo.write(f"  Recall: {metricas['recall']:.4f}\n")
                arquivo.write(f"  F1-Score: {metricas['f1']:.4f}\n")
                arquivo.write(f"  Diferença (Treino - Val): {metricas['diff_overfit']:.4f}\n")
                arquivo.write("\n")
            
            arquivo.write("\n2. ANÁLISE DE OVERFITTING\n")
            arquivo.write("-" * 60 + "\n\n")
            
            for nome, metricas in self.resultados['metricas'].items():
                diff = metricas['diff_overfit']
                if diff > 0.1:
                    status = "ALTO RISCO DE OVERFITTING"
                elif diff > 0.05:
                    status = "RISCO MODERADO DE OVERFITTING"
                else:
                    status = "BAIXO RISCO DE OVERFITTING"
                
                arquivo.write(f"{nome}: {status}\n")
                arquivo.write(f"  Diferença: {diff:.4f}\n\n")
            
            arquivo.write("\n3. MELHOR MODELO\n")
            arquivo.write("-" * 60 + "\n\n")
            
            melhor_acuracia = max(self.resultados['metricas'].items(),key=lambda x: x[1]['acuracia'])
            melhor_generalizacao = min(self.resultados['metricas'].items(),key=lambda x: x[1]['diff_overfit'])
            arquivo.write(f"Melhor Acurácia: {melhor_acuracia[0]}\n")
            arquivo.write(f"  Acurácia: {melhor_acuracia[1]['acuracia']:.4f}\n\n")
            arquivo.write(f"Melhor Generalização: {melhor_generalizacao[0]}\n")
            arquivo.write(f"  Diferença: {melhor_generalizacao[1]['diff_overfit']:.4f}\n")
            arquivo.write("\n4. DISCUSSÃO\n")
            arquivo.write("-" * 60 + "\n\n")
            modelo_base_diff = self.resultados['metricas']['base']['diff_overfit']

            if modelo_base_diff > 0.1:
                arquivo.write("1. A rede base apresentou sinais de overfitting, com diferença ")
                arquivo.write(f"significativa entre treino e validação ({modelo_base_diff:.4f}).\n\n")
            else:
                arquivo.write("1. A rede base não apresentou overfitting significativo.\n\n")
            
            arquivo.write("2. Técnicas de regularização aplicadas:\n")
            arquivo.write("   - Dropout (taxa de 0.3)\n")
            arquivo.write("   - Regularização L2 (peso 0.001)\n")
            arquivo.write("   - Redução de complexidade (menos camadas/neurônios)\n")
            arquivo.write("   - Combinação Dropout + L2\n")
            arquivo.write("   - Early Stopping (aplicado em todos os modelos)\n\n")
            
            arquivo.write("3. Melhoria em dados não vistos:\n")
            melhor_modelo = max(self.resultados['metricas'].items(),key=lambda x: x[1]['acuracia'])
            arquivo.write(f"   O modelo com melhor desempenho em dados não vistos foi ")
            arquivo.write(f"'{melhor_modelo[0]}' com acurácia de {melhor_modelo[1]['acuracia']:.4f}\n\n")
            
            arquivo.write("4. Melhor equilíbrio entre desempenho e generalização:\n")
            scores = {}
            for nome, metricas in self.resultados['metricas'].items():
                score = (0.7 * metricas['acuracia'] + 
                        0.3 * (1 - metricas['diff_overfit']))
                scores[nome] = score
            
            melhor_equilibrio = max(scores.items(), key=lambda x: x[1])
            arquivo.write(f"   '{melhor_equilibrio[0]}' apresentou o melhor equilíbrio ")
            arquivo.write(f"com score de {melhor_equilibrio[1]:.4f}\n")
        
        print(f" Relatório salvo: {self.diretorio / 'relatorio_regularizacao.txt'}")
    
    def _exibir_analise(self):

        print("\n" + "=" * 60)
        print("ANÁLISE DE REGULARIZAÇÃO - RESUMO")
        print("=" * 60)
        
        melhor_acuracia = max(
            self.resultados['metricas'].items(),
            key=lambda x: x[1]['acuracia']
        )
        
        melhor_generalizacao = min(
            self.resultados['metricas'].items(),
            key=lambda x: x[1]['diff_overfit']
        )
        
        print(f"\n Melhor Acurácia: {melhor_acuracia[0]}")
        print(f"   Acurácia: {melhor_acuracia[1]['acuracia']:.4f}")
        
        print(f"\n Melhor Generalização: {melhor_generalizacao[0]}")
        print(f"   Diferença Treino-Val: {melhor_generalizacao[1]['diff_overfit']:.4f}")
        
        # Verificar overfitting
        print("\nStatus de Overfitting:")
        for nome, metricas in self.resultados['metricas'].items():
            diff = metricas['diff_overfit']
            if diff > 0.1:
                status = " ALTO"
            elif diff > 0.05:
                status = "MODERADO"
            else:
                status = " BAIXO"
            
            # Remover "Modelo " do nome
            nome_curto = nome.replace('Modelo ', '').replace('com ', '')
            print(f"   {nome_curto}: {status} (dif={diff:.4f})")
        
        print("\n" + "=" * 60)
        print(" Análise de regularização concluída!")
        print(f" Resultados salvos em: {self.diretorio}")
        print("=" * 60)


def executar_analise_regularizacao(diretorio_base, X_treino, X_validacao, X_teste,y_treino, y_validacao, y_teste):

    diretorio = Path(diretorio_base) / 'regularizacao'
    analise = AnaliseRegularizacao(diretorio)
    analise.executar_analise(X_treino, X_validacao, X_teste,y_treino, y_validacao, y_teste)
    
    return analise