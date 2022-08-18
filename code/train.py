# Importar librerías básicas
import gzip 
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, metrics, cv

# Construímos un diccionario cuyas llaves son los nombres de las columnas relevantes de nuestro conjunto de datos y cuyos valores son el tipo de valor de las celdas de dicha columna
dtype = {'fiIdTienda':'object','fiEvento':'object', 'fiPeriodo':'object', 'SKU':'object', 'Tienda':'object', 'FORMATO':'object','REGION':'object', 'Categoria':'object', 'Proveedor':'object','DEPTO':'object', 'TipoResurtible':'object','Mes':'object', 'Sem':'object', 'DiaSem':'object', 'DiaMes':'object', 'DiaAño':'object', 'Decil':'object', 'Importado':'object','MarcaPropia':'object', 'EnOferta':'object', 'Existencia':'float', 'fiDesplazamiento':'float','VentaPromedio':'float', 'Y_Faltante':'int'}

# Almacenamos los nombres segmentados por el tipo de dato cuya respectiva columna contiene 
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


# Cargamos el conjunto de datos con las columnas relevantes cargadas y bajo una codificación 'latin-1'
file = 'data/train_data_2022.csv'
data = pd.read_csv(file, header=0,low_memory=False,encoding='latin-1',usecols=relevant_columns,dtype=dtype)

# Preprocesamiento de datos 
data = data.dropna()
target = data['Y_Faltante']
yes = target[target == 1].count()
no = target[target == 0].count()
## Balanceamos a 50% - 50% nuestro conjunto de datos.
data_yes = data[data["Y_Faltante"] == 1]
data_no = data[data["Y_Faltante"] == 0]
over_sampling = data_yes.sample(no, replace=True, \
random_state = 0)
balanced_data = pd.concat([data_no, over_sampling], \
axis=0)
data = balanced_data.reset_index(drop=True)

X = data.drop('Y_Faltante',axis=1)
y = data.Y_Faltante

# Se construyen los conjuntos de entrenamiento y prueba sobre los que se aplicarán algoritmos de machine learning. Es importante hacer esta escisión para poder usar el parámetro de usar mejor modelo en el catboost
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.17, random_state=0)

# Aplicamos un min-max scaling sobre las variables numéricas de nuestro conjunto de entrenamiento
categorical = X_train.drop(columns=['Existencia','fiDesplazamiento','VentaPromedio'])
numerical = X_train[['Existencia','fiDesplazamiento','VentaPromedio']]
numerical_scaled = (numerical - numerical.min())/(numerical.max() - numerical.min())
X_train = pd.concat([categorical,numerical_scaled], axis=1)

# Listamos los tipos de variables a ser usados
indices_categoricos = np.where(X_train.dtypes != float)[0]
num_ind = np.where(X_train.dtypes != object)[0]

# Invocamos el algoritmo CatBoost
params = {
    'l2_leaf_reg':int(1.0),
    #'iterations': 2000,
    #'eval_metric': [metrics.Accuracy()],
    'custom_loss':[metrics.Accuracy()],
    'random_seed': 0,
    'task_type':'GPU',
    'logging_level': 'Verbose',
    'use_best_model': True,
    'save_snapshot':True
}
train_pool = X_train, y_train
validate_pool = (X_val, y_val)

params_with_snapshot = params.copy()

model = CatBoostClassifier(**params_with_snapshot)
model.fit(X_train,y_train,cat_features=indices_categoricos, eval_set=(X_val,y_val),logging_level='Verbose',save_snapshot=True,plot=True)

model.save_model('Ago_model2022.dump')
#model = CatBoostClassifier()

