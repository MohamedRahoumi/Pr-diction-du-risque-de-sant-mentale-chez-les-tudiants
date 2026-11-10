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
| `Study_Hours_Per_Week` | Numérique | Heures d'étude par semaine (h) | Médiane + Capping IQR + Standardisation |
| `GPA` | Numérique | Moyenne académique (échelle 0.0 à 4.0) | Médiane + Capping IQR + Standardisation |
| `Social_Media_Hours` | Numérique | Temps quotidien sur les réseaux sociaux (h) | Médiane + Capping IQR + Standardisation |
| `Screen_Time_Hours` | Numérique | Temps d'écran quotidien total (h) | Médiane + Capping IQR + Standardisation |
| `Physical_Activity` | Catégoriel Ordinal | Activité physique (`Low`, `Moderate`, `High`) | Encodage Ordinal [0, 1, 2] |
| `Financial_Stress` | Numérique | Niveau de stress financier (score 1 à 10) | Médiane + Standardisation |
| `Family_Support` | Numérique | Niveau de soutien familial perçu (score 1 à 10) | Médiane + Standardisation |
| `Attendance_Rate` | Numérique | Taux d'assiduité aux cours (%) | Médiane + Capping IQR + Standardisation |
| `Part_Time_Job` | Binaire | Exercice d'un emploi étudiant (`Yes`, `No`) | One-Hot Encoding |
| `Diet_Quality` | Catégoriel Ordinal | Qualité perçue du régime alimentaire (`Poor`, `Average`, `Good`) | Encodage Ordinal [0, 1, 2] |
| `Previous_Mental_Health_History` | Binaire | Antécédents de santé mentale personnels (`Yes`, `No`) | One-Hot Encoding |
| **`Mental_Health_Risk`** | **Cible (3 classes)** | **Niveau de risque global (`Low`, `Medium`, `High`)** | Variable cible à prédire |

---

## 🧭 3. Démarche Méthodologique

Le projet suit rigoureusement les 5 étapes demandées par le brief pédagogique :

```mermaid
flowchart LR
    A["1. EDA<br/>(notebooks/1-eda.ipynb)"] --> B["2. Preprocessing<br/>(notebooks/2-preprocessing.ipynb)"]
    B --> C["3. Entraînement<br/>(notebooks/3-training.ipynb)"]
    C --> D["4. Optimisation<br/>(notebooks/4-optimization.ipynb)"]
    D --> E["5. Déploiement<br/>(Airflow + Streamlit)"]
```

### Étape 1 — Chargement & Analyse Exploratoire (EDA)
* Audit de conformité : structure des types, absence de doublons d'enregistrement, analyse des valeurs manquantes (~0.5 % sur quelques colonnes).
* Analyse de distribution : répartition équilibrée de la cible multi-classes (~33 % pour chaque classe `Low`, `Medium`, `High`).
* Détection statistique des valeurs aberrantes (*outliers*) via la méthode de l'écart interquartile (IQR).
* Analyse de corrélation et d'association (Pearson, Spearman et ANOVA) : mise en évidence d'une relation monotone forte entre le déficit de sommeil, le stress financier élevé, le manque de soutien familial et l'élévation du risque psychologique.

### Étape 2 — Transformation & Préparation des Données
* **Split étanche :** Séparation immédiate train (80 %) / test (20 %) avec stratification afin de garantir l'absence absolue de fuite d'information (*data leakage*).
* **Imputation robuste :** `SimpleImputer` par la médiane pour les variables continues et par le mode (`most_frequent`) pour les variables catégorielles.
* **Traitement des outliers :** Création d'un transformateur scikit-learn sur mesure (`IQRCapper`) calculant les bornes $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$ uniquement sur le jeu d'entraînement.
* **Encodage différentié :**
  * `OneHotEncoder(handle_unknown='ignore')` pour les variables nominales (`Gender`, `Part_Time_Job`, `Previous_Mental_Health_History`).
  * `OrdinalEncoder` respectant la sémantique ordonnée (`Physical_Activity`, `Diet_Quality`).
* **Standardisation :** `StandardScaler` pour aligner les échelles numériques sans biaiser les estimateurs à base de marge ou de gradient.
* **Sérialisation :** Pipeline de prétraitement complet sauvegardé dans `data/processed/preprocessor.pkl`.

### Étape 3 — Entraînement & Comparaison des Modèles
Comparaison sur le même ensemble d'entraînement via validation croisée stratifiée à 5 plis (*Stratified 5-Fold CV*) :
1. **Régression Logistique Multinomiale (Baseline)** : Modèle linéaire régularisé $L_2$.
2. **Support Vector Machine (Linear SVM)** : Séparateur à marge maximale avec noyau linéaire.
3. **Random Forest Classifier** : Ensemble d'arbres décisionnels avec bagging.

### Étape 4 — Optimisation & Évaluation Finale
* **Fine-tuning des hyperparamètres :** Recherche exhaustive sur grille (`GridSearchCV`) sur le paramètre de régularisation $C$ et la perte `hinge`/`squared_hinge`.
* **Calibration de probabilités :** Utilisation de `CalibratedClassifierCV(method='sigmoid')` afin de fournir à l'interface applicative des probabilités prédites fiables pour les 3 classes.
* **Évaluation sur test indépendant (400 étudiants) :** Calcul de l'Accuracy, du Rappel, de la Précision et du F1-score pondéré ainsi que du score ROC-AUC multiclasse (One-vs-Rest).
* **Export de production :** Pipeline complet de bout en bout sérialisé dans `models/mental_health_risk_pipeline.pkl`.

### Étape 5 — Orchestration Airflow & Application Streamlit
* **DAG Apache Airflow (`dags/mental_health_dag.py`) :** Pipeline automatisé composé de 4 tâches séquentielles :
  1. `check_raw_data` : validation de présence du fichier de données brutes.
  2. `preprocess_data` : exécution du pipeline de nettoyage et split train/test.
  3. `train_model` : entraînement du SVM calibré.
  4. `evaluate_model` : vérification des seuils de performance avant mise en production.
* **Application interactive Streamlit (`app/streamlit_app.py`) :** Interface ergonomique comprenant un simulateur de prédiction en temps réel, un tableau de bord analytique des métriques, une exploration interactive du jeu de test et un rappel méthodologique complet.

---

## 📈 4. Résultats et Métriques de Performance

### Comparatif sur le Jeu de Test Indépendant (400 étudiants)

| Modèle | F1-Score (pondéré) | Accuracy | ROC-AUC (ovr) | Précision (pondérée) | Rappel (pondéré) | Décision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Linear SVM (Optimisé + Calibré)** | **76.21 %** | **76.25 %** | **88.08 %** | **76.17 %** | **76.25 %** | **Retenu pour la Production** |
| Logistic Regression (Baseline) | 75.75 % | 75.75 % | 91.50 % | 76.10 % | 75.75 % | Modèle de référence |
| SVM (Baseline brute non optimisée) | 72.14 % | 72.25 % | — | 72.10 % | 72.25 % | Améliorable par tuning |
| Random Forest Classifier | 67.21 % | 67.75 % | 85.28 % | 67.89 % | 67.75 % | Écarté (surapprentissage des frontières) |

### Matrice de Confusion du Modèle Final (400 étudiants)

```text
                     Prédit Low      Prédit Medium      Prédit High
  Réel Low              109               22                 1
  Réel Medium            23               84                25
  Réel High               0               24               112
```

### 🔍 Analyse Qualifiée des Résultats
* **Absence de faux négatifs critiques :** **0 étudiant** à risque élevé (`High`) n'a été classé à tort en risque faible (`Low`). Cette propriété est primordiale sur le plan éthique et préventif.
* **Erreurs adjacentes tolérables :** La quasi-totalité des confusions résiduelles survient entre classes immédiatement contiguës (`Low` $\leftrightarrow$ `Medium` et `Medium` $\leftrightarrow$ `High`).
* **Interprétabilité clinique :** Les variables ayant les poids normalisés les plus déterminants sont la privation de sommeil, le stress financier et l'absence de soutien familial.

