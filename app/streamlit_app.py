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
            "Relance l'entraînement dans cet environnement."
        )


def get_metrics():
    if not METRICS_PATH.exists():
        return None, "Métriques introuvables. Lance le notebook 04 ou le DAG Airflow."
    try:
        return _load_metrics(METRICS_PATH.stat().st_mtime), None
    except (OSError, json.JSONDecodeError):
        return None, "Le fichier de métriques est illisible. Relance l'évaluation."


# --------------------------------------------------------------------------- en-tête
st.title("🎓 Risque de santé mentale étudiant : prototype pédagogique")
st.warning(
    "**Usage strictement pédagogique.** Ce modèle, entraîné sur des données **simulées**, n'est ni un "
    "outil de diagnostic, ni de dépistage clinique, ni de décision disciplinaire, ni de sélection "
    "d'étudiants. Il ne remplace pas l'avis d'un professionnel."
)
tab_perf, tab_test, tab_batch, tab_about = st.tabs(
    ["📊 Performances", "🧪 Tester un profil", "📁 Fichier CSV", "ℹ️ Limites"]
)

model, model_error = get_model()

# --------------------------------------------------------------------------- onglet 1
with tab_perf:
    metrics, err = get_metrics()
    if err:
        st.error(err)
    else:
        st.caption(
            f"Modèle : {metrics.get('model', '?')} · évalué sur {metrics.get('n_test', '?')} "
            "étudiants du jeu de test (jamais vus à l'entraînement)"
        )
        cols = st.columns(5)
        cols[0].metric("F1 (macro)", f"{metrics['f1_macro']:.3f}")
        cols[1].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[2].metric("Précision (macro)", f"{metrics['precision_macro']:.3f}")
        cols[3].metric("Rappel (macro)", f"{metrics['recall_macro']:.3f}")
        cols[4].metric("ROC-AUC (ovr)", f"{metrics['roc_auc_ovr']:.3f}")
        if "cv_f1_macro" in metrics:
            st.caption(f"F1 en validation croisée sur le train : {metrics['cv_f1_macro']:.3f}")

        st.subheader("Matrice de confusion")
        cm = np.array(metrics["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        labels = [fr(c) for c in metrics.get("classes", RISK_ORDER)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[0],
            xticklabels=labels,
            yticklabels=labels,
        )
