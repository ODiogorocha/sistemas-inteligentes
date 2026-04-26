import pandas as pd
import cbrkit as cb
import os

from data_types import DataTypeCases



class DataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return pd.read_csv(self.path)

    def save(self, df):
        df.to_csv(self.path, index=False)

class Case:
    def __init__(self):
        self.path = "/home/diogo/Documentos/codigos/sistemas-inteligentes/Trabalho_1/database/cbr_psychology_110_cases_clinical.csv"
        self.casebase = cb.loaders.file(self.path)
        self.data_types_cases = DataTypeCases()

    def old_case_print(self):
        for lines in self.casebase:
            print(self.casebase)

    def cases_to_list(self):
        list_case = self.casebase
        list_case = list_case.values.tolist()

        return list_case


    def get_user_input_case(self):
        if not os.path.exists(self.path):
            print("Arquivo não encontrado!")
            return None

        df = pd.read_csv(self.path)
        columns = df.columns.tolist()

        new_case = {}

        print("\n=== Inserir novo caso ===")

        next_id_number = len(df) + 1
        auto_id = f"C{next_id_number:03d}"

        for column in columns:
            if column.lower() == "case_id":
                print(f"{column}: {auto_id} (gerado automaticamente)")
                new_case[column] = auto_id
            else:
                value = input(f"{column}: ")
                new_case[column] = value

        return new_case


    def add_new_case_interactive(self):
        """
        Lê o CSV, pede os dados ao usuário e adiciona uma nova linha.
        """
        if not os.path.exists(self.path):
            print("Arquivo não encontrado!")
            return

        df = self.casebase

        columns = self.cases_to_list()

        # Pede dados ao usuário
        new_case = self.get_user_input_case(columns)

        # Converte para DataFrame
        new_row = pd.DataFrame([new_case])

        # Adiciona ao dataset
        df = pd.concat([df, new_row], ignore_index=True)

        # Salva
        df.to_csv(self.path, index=False)
kk
        print("\n Novo caso adicionado com sucesso!")

    def get_old_case(self):
        keyword = input("Digite a palavra chave: ").lower()

        results = self.df[self.df.apply(
            lambda row: keyword in str(row).lower(), axis=1
        )]

        print(results)
