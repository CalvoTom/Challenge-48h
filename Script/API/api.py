from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import joblib

# Chemin absolu vers le dossier contenant api.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin absolu vers le modèle
model_path = os.path.join(BASE_DIR, '..', '..', 'Data', 'Model', 'best_model.joblib')
model = joblib.load(model_path)

app = FastAPI()

# Définir la structure attendue des données d'entrée
class InputData(BaseModel):
    temperature: float
    humidite: float
    force_moyenne_du_vecteur_de_vent: float
    force_du_vecteur_de_vent_max: float
    pluie_intensite_max: float
    sismicite: float
    concentration_gaz: float
    pluie_totale: float
    Month: int
    Day: int
    quartier_1: int
    quartier_2: int
    quartier_3: int
    quartier_4: int
    quartier_5: int

@app.post("/predict")
def predict(data: InputData):
    # Convertir en array numpy
    X = np.array([[
        data.temperature,
        data.humidite,
        data.force_moyenne_du_vecteur_de_vent,
        data.force_du_vecteur_de_vent_max,
        data.pluie_intensite_max,
        data.sismicite,
        data.concentration_gaz,
        data.pluie_totale,
        data.Month,
        data.Day,
        data.quartier_1,
        data.quartier_2,
        data.quartier_3,
        data.quartier_4,
        data.quartier_5
    ]])
    
    pred = model.predict(X)
    return {"prediction": pred[0]}
