import pandas as pd
from ms import model

# Especificar columnas
dtype = {'SKU':'object','FORMATO':'object','REGION':'object','Tienda':'object','Categoria':'object','Proveedor':'object','CLASE':'object','SCLASE':'object','Mes':'object','DiaSem':'object','Sem':'object','Existencia':'float','fiDesplazamiento':'float','VentaPromedio':'float','Y_Faltante':'int'}

relevant_columns = ['fiIdTienda', 'fiEvento', 'fiPeriodo', 'SKU', 'Tienda', 'FORMATO',
       'REGION', 'Categoria', 'Proveedor', 'DEPTO', 'TipoResurtible',
       'Mes', 'Sem', 'DiaSem', 'DiaMes', 'DiaAño', 'Decil', 'Importado',
       'MarcaPropia', 'EnOferta', 'Existencia', 'fiDesplazamiento',
       'VentaPromedio', 'Y_Faltante']

cat_columns = ['fiIdTienda', 'fiEvento', 'fiPeriodo', 'SKU', 'Tienda', 'FORMATO',
       'REGION', 'Categoria', 'Proveedor', 'DEPTO', 'TipoResurtible',
       'Mes', 'Sem', 'DiaSem', 'DiaMes', 'DiaAño', 'Decil', 'Importado',
       'MarcaPropia', 'EnOferta']

num_columns = ['Existencia','fiDesplazamiento','VentaPromedio']
target = ['Y_Faltante']



def predict(Y, model):
    Z = Y[['fiIdTienda', 'fiEvento', 'fiPeriodo', 'SKU', 'Tienda', 'FORMATO',
       'REGION', 'Categoria', 'Proveedor', 'DEPTO', 'TipoResurtible',
       'Mes', 'Sem', 'DiaSem', 'DiaMes', 'DiaAño', 'Decil', 'Importado',
       'MarcaPropia', 'EnOferta', 'Existencia', 'fiDesplazamiento',
       'VentaPromedio', 'Y_Faltante']].copy()
    prediction = model.predict(Z)[0]
    return prediction

def formato_de_tipos(frame):
    dtype = {'SKU':'object','FORMATO':'object','REGION':'object','Tienda':'object','Categoria':'object','Proveedor':'object','CLASE':'object','SCLASE':'object','Mes':'object','DiaSem':'object','Sem':'object','Existencia':'float','fiDesplazamiento':'float','VentaPromedio':'float','Y_Faltante':'int'}
    for n in range(len(dtype)):
        llaves = list(dtype.keys())
        valores = list(dtype.values())
        llave = llaves[n]
        valor = valores[n]
        frame[llave]=frame[llave].astype(valor)
    return frame

# Falta la codificación!!!
def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    Y = formato_de_tipos(X)
    Z = Y.astype(dtype,copy=True,errors='raise')
    prediction = predict(Z, model)
    if prediction == 1:
        label = "Probable Faltante de Anaquel"
    else:
        label = "Producto Encontrado"
    return {
        'label': label,
        'prediction': int(prediction)
    }
