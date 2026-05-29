import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET = os.path.join(
    PROJECT_DIR,
    "db",
    "sim_cnv_obt10br142742187_60_99_223.csv"
)

GRAFICOS = os.path.join(PROJECT_DIR, "graficos")

os.makedirs(GRAFICOS, exist_ok=True)