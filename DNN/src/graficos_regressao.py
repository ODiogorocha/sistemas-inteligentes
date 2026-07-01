import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class GraficosRegressao:
    
    @staticmethod
    def plotar_valores_reais_vs_preditos(y_real, y_predito, diretorio):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_real, y_predito, alpha=0.5, s=30)
        
        min_val = min(y_real.min(), y_predito.min())
        max_val = max(y_real.max(), y_predito.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
        
        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Valores Preditos')
        ax.set_title('Valores Reais vs Valores Preditos')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(diretorio / 'reais_vs_preditos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico reais vs preditos salvo em: {diretorio / 'reais_vs_preditos.png'}")
    
    @staticmethod
    def plotar_residuos(y_real, y_predito, diretorio):
        residuos = y_real - y_predito
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.scatter(y_predito, residuos, alpha=0.5, s=30)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Valores Preditos')
        ax1.set_ylabel('Resíduos')
        ax1.set_title('Resíduos vs Valores Preditos')
        ax1.grid(alpha=0.3)
        
        ax2.hist(residuos, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Resíduos')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição dos Resíduos')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(diretorio / 'analise_residuos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Análise de resíduos salva em: {diretorio / 'analise_residuos.png'}")
    
    @staticmethod
    def plotar_curva_aprendizado(historico, diretorio):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(historico.history['loss'], label='Treino', linewidth=2)
        ax1.plot(historico.history['val_loss'], label='Validação', linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Evolução da Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(historico.history['mae'], label='Treino', linewidth=2)
        ax2.plot(historico.history['val_mae'], label='Validação', linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.set_title('Evolução do MAE')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle('Curvas de Aprendizado - Regressão')
        plt.tight_layout()
        plt.savefig(diretorio / 'curva_aprendizado_regressao.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Curva de aprendizado salva em: {diretorio / 'curva_aprendizado_regressao.png'}")