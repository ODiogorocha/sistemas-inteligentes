import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

COLUNAS_ZERO_INVALIDO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

def tratar_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df.copy()
    for col in COLUNAS_ZERO_INVALIDO:
        n_zeros = (df[col] == 0).sum()
        if n_zeros > 0: 
            mediana = df[col][df[col] != 0].median()
            df[col] = df[col].replace(0, mediana)

            print(f"    {col}: {n_zeros} zeros → substituídos por mediana ({mediana:.2f})")

        return df

def tratar_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower, upper)
            print(f"    {col}: {n_outliers} outliers limitados [{lower:.2f}, {upper:.2f}]")
    return df

def normalizar(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    print(f"    Normalização aplicada (StandardScaler) → shape: {X.shape}")

    return X

def preprocessar(df: pd.DataFrame):
    
    print("\n  → Separando rótulos (Outcome)...")
    rotulos = df["Outcome"].values

    print("  → Tratando zeros inválidos...")
    df_clean = tratar_zeros(df)

    print("  → Tratando outliers (IQR clipping)...")
    df_clean = tratar_outliers_iqr(df_clean)

    print("  → Verificando valores faltantes...")
    nans = df_clean[FEATURES].isnull().sum().sum()
    print(f"    Total de NaN: {nans}")

    print("  → Normalizando...")
    X = normalizar(df_clean)

    print("  → Reduzindo com PCA para visualização...")
    X_pca = aplicar_pca(X, n_componentes=2)

    return X, X_pca, rotulos

def aplicar_pca(X: np.ndarray, n_componentes: int = 2) -> np.ndarray:
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X)
    variancia = pca.explained_variance_ratio_.sum() * 100
    print(f"    PCA 2D aplicado → variância explicada: {variancia:.1f}%")

    return X_pca
