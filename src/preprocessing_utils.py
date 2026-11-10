"""Utilitaires de prétraitement pour le projet de prédiction du risque de santé mentale.

Ce module contient le transformateur custom `IQRCapper`, défini dans un fichier
(et non dans le notebook) pour que la sérialisation joblib/pickle fonctionne
lors du rechargement du pipeline (app Streamlit, DAG Airflow, etc.).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorisation (clipping) des valeurs aberrantes selon la règle de l'IQR.

    À l'`fit` (sur le train uniquement), on calcule pour chaque colonne les
