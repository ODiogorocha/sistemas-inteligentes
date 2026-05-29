from sklearn.model_selection import train_test_split

from config.settings import DATASET, GRAFICOS

from data.data_loader import DataLoader
from data.preprocessing import Preprocessing
from data.descriptive_analysis import DescriptiveAnalysis

from models.decision_tree_model import DecisionTreeModel

from visualization.plot_generator import PlotGenerator


print("=" * 80)
print("TRABALHO PRÁTICO - ÁRVORES DE DECISÃO".center(80))
print("=" * 80)

loader = DataLoader(DATASET)

df = loader.load_data()

print("\nDataset carregado com sucesso!")

print(df.head())

df = Preprocessing.clean_data(df)

df = Preprocessing.create_target(df)

DescriptiveAnalysis.show_general_info(df)

top10 = DescriptiveAnalysis.show_top_diseases(df)

PlotGenerator.plot_top10(top10, GRAFICOS)

X = df.drop(
    columns=[
        "Categoria CID-10",
        "Total",
        "high_mortality"
    ]
)

y = df["high_mortality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

tree_model = DecisionTreeModel(
    X_train,
    X_test,
    y_train,
    y_test
)

configs = [
    {
        "max_depth": 3,
        "criterion": "gini",
        "min_samples_leaf": 1,
        "min_samples_split": 2
    },
    {
        "max_depth": 5,
        "criterion": "gini",
        "min_samples_leaf": 1,
        "min_samples_split": 2
    },
    {
        "max_depth": 10,
        "criterion": "gini",
        "min_samples_leaf": 1,
        "min_samples_split": 2
    }
]

for i, cfg in enumerate(configs):

    model, acc, prec, rec, y_pred = tree_model.evaluate_model(
        cfg["max_depth"],
        cfg["criterion"],
        cfg["min_samples_leaf"],
        cfg["min_samples_split"]
    )

    print("\n" + "-" * 60)
    print(f"MODELO {i + 1}")
    print("-" * 60)

    print(f"Acurácia : {acc}")
    print(f"Precisão : {prec}")
    print(f"Recall   : {rec}")

    top3 = (
        df[["Categoria CID-10", "Total"]]
        .sort_values(by="Total", ascending=False)
        .head(3)
    )

    print("\nTop 3 doenças que mais matam:\n")

    for _, row in top3.iterrows():

        print(
            f"{row['Categoria CID-10']} -> "
            f"{int(row['Total'])} óbitos"
        )

tree_model.show_confusion_matrix(y_test, y_pred)

print("\nExecução finalizada com sucesso!")