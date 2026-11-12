import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import MODEL_PATH, METRICS_PATH, RISK_ORDER, FEATURE_COLS
from src.inference import CATEGORIES, predict_dataframe, validate_dataframe

MAX_ROWS = 10_000
FR = {
    "Low": "Faible",
    "Medium": "Moyen",
    "High": "Élevé",
    "Moderate": "Modérée",
    "Poor": "Mauvaise",
    "Average": "Moyenne",
    "Good": "Bonne",
    "No": "Non",
    "Yes": "Oui",
    "Female": "Femme",
    "Male": "Homme",
    "Non-binary": "Non-binaire",
}
fr = lambda v: FR.get(v, v)

st.set_page_config(page_title="Risque étudiant - prototype", page_icon="🎓", layout="wide")


@st.cache_resource
def _load_model(mtime: float):
    return joblib.load(MODEL_PATH)


@st.cache_data
def _load_metrics(mtime: float):
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_model():
    if not MODEL_PATH.exists():
        return None, (
            "Modèle introuvable. Lance le notebook 04 ou déclenche le DAG Airflow "
            "`mental_health_risk_pipeline_dag` pour le créer."
        )
    try:
        return _load_model(MODEL_PATH.stat().st_mtime), None
    except Exception as exc:
        return None, (
            f"Impossible de charger le modèle ({type(exc).__name__}). Cause fréquente : "
            "version de scikit-learn différente de celle de l'entraînement. "
