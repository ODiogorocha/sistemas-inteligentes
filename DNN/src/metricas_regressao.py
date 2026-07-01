import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricasRegressao:
    
    @staticmethod
    def calcular_todas(y_real, y_predito):

        mse = mean_squared_error(y_real, y_predito)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_real, y_predito)
        r2 = r2_score(y_real, y_predito)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    @staticmethod
    def salvar_metricas(metricas, caminho_arquivo):
        with open(caminho_arquivo, 'w') as arquivo:
            arquivo.write("=" * 60 + "\n")
            arquivo.write("MÉTRICAS DE REGRESSÃO\n")
            arquivo.write("=" * 60 + "\n\n")
            
            for chave, valor in metricas.items():
                arquivo.write(f"{chave}: {valor:.4f}\n")