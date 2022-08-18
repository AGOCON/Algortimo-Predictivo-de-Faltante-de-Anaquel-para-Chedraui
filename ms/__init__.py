# Imports
from fastapi import FastAPI
import joblib
from catboost import CatBoostClassifier, Pool

# Initialize FastAPI app
app = FastAPI()

# Load model
model = CatBoostClassifier()
model = model.load_model('model/Ago_model2022.dump')
