from data_loader import carregar_dados
from analise_descritiva import analise_univariada, analise_bivariada, analise_multivariada
from preprocessamento import preprocessar
from clustering import rodar_kmeans, rodar_agglomerative, rodar_dbscan
from avaliacao import avaliar_todos, exibir_tabela_resultados
from visualizacao import plotar_clusters_pca


def main():
    print("=" * 60)
    print("  CLUSTERING - DATASET DIABETES")
    print("=" * 60)

    # 1. Carregamento
    df = carregar_dados("../data/diabetes.csv")

    # 2. Análise Descritiva
    print("\n[1/5] Análise Descritiva...")
    analise_univariada(df)
    analise_bivariada(df)
    analise_multivariada(df)

    # 3. Pré-processamento
    print("\n[2/5] Pré-processamento...")
    X, X_pca, rotulos = preprocessar(df)

    # 4. Clustering
    print("\n[3/5] Rodando algoritmos de clustering...")
    resultados_kmeans      = rodar_kmeans(X)
    resultados_agglomerative = rodar_agglomerative(X)
    resultados_dbscan      = rodar_dbscan(X)

    todos_resultados = resultados_kmeans + resultados_agglomerative + resultados_dbscan

    # 5. Avaliação
    print("\n[4/5] Avaliando resultados...")
    todos_resultados = avaliar_todos(todos_resultados, X)
    exibir_tabela_resultados(todos_resultados)

    # 6. Visualização
    print("\n[5/5] Gerando visualizações dos melhores clusters...")
    plotar_clusters_pca(todos_resultados, X_pca, rotulos)

    print("\n✓ Concluído! Verifique a pasta 'figuras/' para os gráficos gerados.")


if __name__ == "__main__":
    main()