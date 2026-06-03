import pandas as pd

def carregar_dados(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)

    print(f"\nDataset carregado: {caminho}")
    print(f"  Instâncias : {df.shape[0]}")
    print(f"  Colunas    : {df.shape[1]}")
    print(f"\nColunas e tipos:")

    for col, dtype in df.dtypes.items():
        print(f"  {col:<30} {dtype}")
    
    print(f"\nPrimeiras linhas:")
    print(df.head())

    return df
