import pandas as pd


class CBRKitSystem:
    def __init__(self, path):
        self.df = pd.read_csv(path)

        # transforma em lista de casos
        self.casebase = self.df.to_dict(orient="records")

    # função de similaridade
    def similarity(self, case, query):
        score = 0

        try:
            # 🔥 categóricos (com peso)
            if case.get("main_issue") == query.get("main_issue"):
                score += 4

            if case.get("clinical_severity") == query.get("clinical_severity"):
                score += 3

            if case.get("stress_level") == query.get("stress_level"):
                score += 2

            #  numéricos
            score += 1 / (1 + abs(case.get("anxiety_score", 0) - query.get("anxiety_score", 0)))
            score += 1 / (1 + abs(case.get("depression_score", 0) - query.get("depression_score", 0)))

        except Exception:
            return 0

        return score

    def retrieve(self, query, k=3):
        scored_cases = []

        for case in self.casebase:
            sim = self.similarity(case, query)
            scored_cases.append((case, sim))

        # ordena por similaridade (maior primeiro)
        scored_cases.sort(key=lambda x: x[1], reverse=True)

        return scored_cases[:k]

    # adaptação
    def adapt(self, cases):
        solutions = [
            c.get("recommendation_text")
            for c in cases
            if c.get("recommendation_text") is not None
        ]

        if not solutions:
            return "Nenhuma recomendação encontrada"

        return max(set(solutions), key=solutions.count)

    # ciclo completo
    def solve(self, query):
        retrieved = self.retrieve(query)

        # separa casos e scores
        retrieved_cases = [case for case, _ in retrieved]

        solution = self.adapt(retrieved_cases)

        return solution, retrieved