from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "student_mental_health.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Modèle sérialisé : vérification du chemin présent
if (ROOT_DIR / "models" / "mental_health_risk_pipeline.pkl").exists():
    MODEL_PATH = ROOT_DIR / "models" / "mental_health_risk_pipeline.pkl"
else:
    MODEL_PATH = ROOT_DIR / "models" / "pipeline.joblib"

METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
TRAIN_INFO_PATH = ROOT_DIR / "models" / "train_info.json"
FIG_DIR = ROOT_DIR / "reports" / "figures"
