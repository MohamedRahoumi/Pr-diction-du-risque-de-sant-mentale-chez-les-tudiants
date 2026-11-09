# 🧠 Prédiction du Risque de Santé Mentale chez les Étudiants

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.10-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white)](https://airflow.apache.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
---

---

## 📌 1. Contexte du Projet & Objectifs Métier

Projet Data Science réalisé dans le cadre du **Brief Jury Blanc (YouCode)**.  
L'objectif est d'étudier les facteurs associés aux difficultés psychologiques chez les étudiants, de concevoir un pipeline rigoureux de Machine Learning supervisé (classification multi-classes : `Low`, `Medium`, `High`), et de déployer une solution complète comprenant une orchestration automatisée avec **Apache Airflow** et une interface d'aide à la décision interactive avec **Streamlit**.

### 🎯 Objectifs Métier
* **Détection précoce des risques :** Identifier les étudiants présentant un risque accru de détresse psychologique.
* **Prévention ciblée :** Anticiper les besoins d'accompagnement avant l'aggravation des symptômes ou le décrochage académique.
* **Compréhension des facteurs déterminants :** Mettre en évidence les leviers d'action (sommeil, stress financier, soutien familial, équilibre vie-études).
* **Orientation & Accompagnement :** Faciliter la mise en relation bienveillante avec les cellules d'écoute et les services sociaux universitaires.
* **Priorisation des ressources :** Optimiser l'allocation des effectifs d'aide médico-sociale sur le campus.

> ⚠️ **Avertissement éthique et légal (Strictement Pédagogique) :**  
> Ce système constitue un prototype d'aide à l'orientation et à la prévention précoce basé sur des données synthétiques.  
> **Il ne doit en aucun cas être utilisé comme un outil de diagnostic médical, de dépistage clinique, de sélection d'étudiants, d'évaluation académique ou de décision disciplinaire.**

---

## 📊 2. Jeu de Données (Dataset)

* **Source :** [Mental Health Risk Prediction Dataset — Students (Kaggle)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
* **Emplacement local :** `data/raw/student_mental_health.csv`
* **Dimensions :** 2 000 observations, 17 variables (16 caractéristiques explicatives + 1 cible)
* **Partitionnement :** Split strict 80 % entraînement (1 600 lignes) / 20 % test indépendant (400 lignes), stratifié sur la cible.

### Dictionnaire des variables

| Variable | Type | Description | Traitement dans le Pipeline |
|---|---|---|---|
| `Student_ID` | Identifiant | Identifiant unique de l'étudiant | Supprimé (aucune valeur prédictive) |
| `Age` | Numérique | Âge de l'étudiant (18–35 ans) | Médiane + Standardisation |
| `Gender` | Catégoriel | Genre (`Male`, `Female`, `Other`) | One-Hot Encoding |
| `Year_of_Study` | Numérique | Année du cursus universitaire (1 à 5) | Médiane + Standardisation |
| `Sleep_Hours` | Numérique | Heures de sommeil moyen par nuit (h) | Médiane + Capping IQR + Standardisation |
