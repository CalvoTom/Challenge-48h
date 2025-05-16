# 🚀 48H Prediction Challenge

Ce projet consiste à entraîner un modèle capable de prédire une catastrophe environemental à partir de données météorologique, temporelle et géographique. Une API est également mise en place pour interroger le modèle entrainer de façon simple.

---

## 📂 Structure du projet

📁 48H/

├── 📁 Data/

│   ├── 📁 Model/

│   │   └── 📄 best_model.joblib  # Modèle déjà entrainé

│   ├── 📁 Preprocessed/

│   │   └── 📄 catastrophes_naturelles_preprocessed.csv  # Données csv Nétoyé et normalisé

│   ├── 📁 Raw/

│   │   └── 📄 catastrophes_naturelles.csv  # Données brut

├── 📁 Script/

│   ├── 📁 Api/

│   │   └── 📄 api.py           # Code de l'API FastAPI

│   ├── 📁 Notebooks/

│   │   └── 📄 Data_preprocessing.ipynb           # Noteboock de nettoyage et de normalisation des données

│   │   └── 📄 Data_processing.ipynb           # Noteboock d'entrainement et d'évaluation du modèle

├── 📄 requirements.txt

└── 📄 README.md

---

## 🧠 Modèle utilisé

Le modèle sélectionné est un GradientBoostingClassifier avec les meilleurs hyperparamètres trouvés via un RandomizedSearchCV.

Hyperparamètres optimaux :

{
  model: GradientBoostingClassifier(random_state=42),
  model__n_estimators: 200,
  model__min_samples_split: 2,
  model__min_samples_leaf: 4,
  model__max_depth: 3,
  model__learning_rate: 0.01
}

---

## 📊 Visualisations

Le projet inclut :

- Matrice de corrélation

- Feature importance

- Courbes ROC multiclass

- Importance des caractéristiques

---

## 🔧 Installation

1. Cloner le repo
```bash
git clone https://github.com/CalvoTom/48H.git
```
2. Créer un environnement virtuel (via conda)
```bash
conda create -n myEnv python=3.12
conda activate myEnv
```
3. Installer les dépendances
```bash
pip install -r requirements.txt
```
---

## 🚀 Lancer l’API

1. Assure-toi que le modèle best_model.joblib est dans Data/Model/

2. Lancer FastAPI avec Uvicorn :
```bash
cd Script/Api
uvicorn api:app --reload
```

3. Ouvrir dans le navigateur :

http://127.0.0.1:8000/docs

## 👨‍💻 Auteur

[CalvoTom](https://github.com/CalvoTom)
