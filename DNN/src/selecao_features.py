import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif


class SelecaoFeatures:

    def __init__(self):
        self.feature_importance = None
        self.features_selecionadas = None
    
    def selecionar_por_importancia_random_forest(self, X_treino, X_teste, y_treino,nomes_features, quantidade_features=10):

        print("\n Selecionando features com Random Forest...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_treino, y_treino)
        
        self.feature_importance = rf.feature_importances_
        
        importancia_df = pd.DataFrame({
            'feature': nomes_features,
            'importancia': self.feature_importance
        }).sort_values('importancia', ascending=False)
        
        self.features_selecionadas = importancia_df.head(quantidade_features)['feature'].tolist()
        
        print(f" {len(self.features_selecionadas)} features selecionadas")
        
        indices_selecionados = [nomes_features.index(f) for f in self.features_selecionadas]
        X_treino_selecionado = X_treino[:, indices_selecionados]
        X_teste_selecionado = X_teste[:, indices_selecionados]
        
        return X_treino_selecionado, X_teste_selecionado, self.features_selecionadas
    
    def selecionar_por_mutual_information(self, X_treino, y_treino, nomes_features,quantidade_features=10):

        print("\n Selecionando features com Mutual Information...")
        
        mi = mutual_info_classif(X_treino, y_treino, random_state=42)
        
        importancia_df = pd.DataFrame({
            'feature': nomes_features,
            'importancia': mi
        }).sort_values('importancia', ascending=False)
        
        self.features_selecionadas = importancia_df.head(quantidade_features)['feature'].tolist()
        self.feature_importance = mi
        
        print(f" {len(self.features_selecionadas)} features selecionadas")
        
        return self.features_selecionadas
    
    def selecionar_por_anova(self, X_treino, y_treino, nomes_features,quantidade_features=10):

        print("\n Selecionando features com ANOVA F-test...")
        
        selector = SelectKBest(f_classif, k=quantidade_features)
        selector.fit(X_treino, y_treino)
        
        self.feature_importance = selector.scores_
        
        indices_selecionados = selector.get_support(indices=True)
        self.features_selecionadas = [nomes_features[i] for i in indices_selecionados]
        
        print(f" {len(self.features_selecionadas)} features selecionadas")
        
        return self.features_selecionadas
    
    def plotar_importancia(self, diretorio, top_n=15):
        if self.feature_importance is None:
            raise ValueError("Nenhuma importância calculada. Execute seleção primeiro.")
        
        importancia_df = pd.DataFrame({
            'feature': self.features_selecionadas,
            'importancia': self.feature_importance[:len(self.features_selecionadas)]
        }).sort_values('importancia', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_features = importancia_df.tail(top_n)
        
        ax.barh(top_features['feature'], top_features['importancia'])
        ax.set_xlabel('Importância')
        ax.set_title(f'Top {top_n} Features Mais Importantes')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(diretorio / 'importancia_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Gráfico de importância salvo em: {diretorio / 'importancia_features.png'}")
    
    def salvar_features_selecionadas(self, caminho_arquivo):

        with open(caminho_arquivo, 'w') as arquivo:
            arquivo.write("=" * 60 + "\n")
            arquivo.write("FEATURES SELECIONADAS\n")
            arquivo.write("=" * 60 + "\n\n")
            arquivo.write(f"Total: {len(self.features_selecionadas)} features\n\n")
            for i, feature in enumerate(self.features_selecionadas, 1):
                arquivo.write(f"{i:3d}. {feature}\n")