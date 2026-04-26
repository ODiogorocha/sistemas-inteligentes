import requests
import numpy as np

from cbrkit.loaders import load_csv
from cbrkit.sim.attribute_value import attribute_value
from cbrkit.sim.numbers import linear
from cbrkit.sim.strings import levenshtein
from cbrkit.retrieval import retrieve

from case_model import Case



class Behaviors:
    def __init__(self):
        self.database = Case()
    
    def define_similarity(self, case_x: str, case_y: str) -> float:
        case_data = load_csv(self.database.path)

        similarity = attribute_value({

            
            "age": linear(max_distance=100),
            "anxiety_score": linear(max_distance=100),
            "depression_score": linear(max_distance=100),
            "stress_level": linear(max_distance=100),
            "sleep_quality": linear(max_distance=100),
            "sleep_hours": linear(max_distance=100),
            "symptom_duration_months": linear(max_distance=100),
            "gad7_estimate": linear(max_distance=100),
            "phq9_estimate": linear(max_distance=100),
            "irritability_level": linear(max_distance=100),
            "work_or_study_impairment": linear(max_distance=100),
            "bmi_estimate": linear(max_distance=100),

            
            "gender": levenshtein(),
            "social_support": levenshtein(),
            "physical_activity": levenshtein(),
            "panic_symptoms": levenshtein(),
            "concentration_difficulty": levenshtein(),
            "appetite_change": levenshtein(),
            "prior_treatment": levenshtein(),
            "current_medication": levenshtein(),
            "trauma_history": levenshtein(),
            "substance_use_risk": levenshtein(),
            "comorbid_profile": levenshtein(),
            "clinical_severity": levenshtein(),

            "main_issue": levenshtein(),
        })

    def embendding_func(self, text: str):
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )

        if response.status_code != 200:
            raise Exception("Erro ao chamar Ollama")

        data = response.json()
        return np.array(data["embedding"])
    
    def embed_func(self,text: str):
        return self.embendding_func(text)


    def ollama_embedding(self):
        """
        Retorna função de embedding para usar no CBRkit
        """
        self.embed_func()
        return embed_func()