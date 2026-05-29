from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


class DecisionTreeModel:

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def evaluate_model(
        self,
        max_depth,
        criterion,
        min_samples_leaf,
        min_samples_split
    ):

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=42
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)

        return (
            model,
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            y_pred
        )

    @staticmethod
    def show_confusion_matrix(y_test, y_pred):

        print("\nMatriz de confusão:\n")
        print(confusion_matrix(y_test, y_pred))

        print("\nRelatório de classificação:\n")
        print(classification_report(y_test, y_pred))