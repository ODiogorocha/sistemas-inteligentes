import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class Metricas:
    
    @staticmethod
    def calcular_todas(y_real, y_predito, y_proba=None):
        
        metricas = {
            'accuracy': accuracy_score(y_real, y_predito),
            'precision': precision_score(y_real, y_predito, average='binary'),
            'recall': recall_score(y_real, y_predito, average='binary'),
            'f1_score': f1_score(y_real, y_predito, average='binary')
        }
        
        
        if y_proba is not None:
            try:
                metricas['roc_auc'] = roc_auc_score(y_real, y_proba)
            except:
                metricas['roc_auc'] = 0.0
        
        metricas['confusion_matrix'] = confusion_matrix(y_real, y_predito)
        
        return metricas
    
    @staticmethod
    def salvar_metricas(metricas, caminho_arquivo):

        with open(caminho_arquivo, 'w') as arquivo:
            arquivo.write("=" * 60 + "\n")
            arquivo.write("MÉTRICAS DE CLASSIFICAÇÃO\n")
            arquivo.write("=" * 60 + "\n\n")
            
            for chave, valor in metricas.items():
                if chave == 'confusion_matrix':
                    arquivo.write("Matriz de Confusão:\n")
                    arquivo.write(str(valor) + "\n\n")
                else:
                    arquivo.write(f"{chave}: {valor:.4f}\n")