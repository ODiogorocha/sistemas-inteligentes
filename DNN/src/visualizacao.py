import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix



class Visualizacao:
    
    @staticmethod
    def plotar_curva_aprendizado(historico, diretorio, nome_experimento=""):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(historico.history['loss'], label='Treino', linewidth=2)
        ax1.plot(historico.history['val_loss'], label='Validação', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Evolução da Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(historico.history['accuracy'], label='Treino', linewidth=2)
        ax2.plot(historico.history['val_accuracy'], label='Validação', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.set_title('Evolução da Acurácia')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Curvas de Aprendizado - {nome_experimento}')
        plt.tight_layout()
        
        nome_arquivo = f'curva_aprendizado_{nome_experimento.replace(" ", "_")}.png'
        plt.savefig(diretorio / nome_arquivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Curva de aprendizado salva em: {diretorio / nome_arquivo}")
    
    @staticmethod
    def plotar_matriz_confusao(y_real, y_predito, diretorio, nome_experimento=""):

        matriz = confusion_matrix(y_real, y_predito)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_title(f'Matriz de Confusão - {nome_experimento}')
        plt.tight_layout()
        nome_arquivo = f'matriz_confusao_{nome_experimento.replace(" ", "_")}.png'
        plt.savefig(diretorio / nome_arquivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Matriz de confusão salva em: {diretorio / nome_arquivo}")
    
    @staticmethod
    def plotar_curva_roc(y_real, y_proba, diretorio, nome_experimento=""):

        fpr, tpr, _ = roc_curve(y_real, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.set_title(f'Curva ROC - {nome_experimento}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        nome_arquivo = f'curva_roc_{nome_experimento.replace(" ", "_")}.png'
        plt.savefig(diretorio / nome_arquivo, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Curva ROC salva em: {diretorio / nome_arquivo}")