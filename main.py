
# Local imports
import datetime

# Third party imports
from pydantic import BaseModel, Field

from ms import app
from ms.functions import get_model_response


model_name = "Prueba de Algortimo Faltante de Anaquel"
version = "v1.0.0"


# Input for data validation
class Input(BaseModel):
    SKU: object = Field(...)#, gt=0)
    Tienda: object = Field(...)#, gt=0)
    FORMATO: object = Field(...)#, gt=0)
    Categoria: object = Field(...)#, gt=0)
    CLASE: object = Field(...)#, gt=0)
    DiaSem: object = Field(...)#, gt=0)
    Sem: object = Field(...)#, gt=0)
    Existencia: float = Field(...)#, gt=0)
    Desplazamiento: float = Field(...)#, gt=0)

    class Config:
        schema_extra = {
            "SKU": 3001984,
            "Tienda": 719,
            "FORMATO": 9,
            "Categoria": 1006006006,
            "CLASE": 6,
            "SCLASE": 6,
            "DiaSem":7,
            "Sem":6,
            "Existencia":22.0,
            "Desplazamiento":46
        }


# Ouput for data validation
class Output(BaseModel):
    label: str
    prediction: int


@app.get('/')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Predicción de Ago Consultores"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response

