####### Este script genera un reporte para Formato completo. 

# Hay que analizar el tema de la normalización antes de encajar los datos categoricos.
# This is the first implementation
import math
import numpy as np
import pandas as pd
import keras_tuner
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import seaborn as sns
from fastai.tabular.all import *

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Accuracy, Precision, Recall 

from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score




seed = 0 

file = 'Datos_faltante_8semanas_20220606 (balanceado para entrenar).csv'
data = pd.read_csv(file, header=0,low_memory=True)#, usecols=data_columns)
print('rows:', data.shape[0], ' columns:', data.shape[1])

# Aislamos las variables numéricas y realizamos un min-max scaling sobre ellas
categorical = data.drop(columns=['Existencia','fiDesplazamiento','VentaPromedio','Y_Faltante'])
numerical = data[['Existencia','fiDesplazamiento','VentaPromedio']]
#numerical_scaled = (numerical - numerical.min())/(numerical.max() - numerical.min()) 
faltante = data["Y_Faltante"]

Existencia = numerical['Existencia']
Desplazamiento = numerical['fiDesplazamiento']
VentaPromedio = numerical['VentaPromedio']

Existencia = pd.to_numeric(Existencia,errors='coerce')
Desplazamiento = pd.to_numeric(Desplazamiento,errors='coerce')
VentaPromedio = pd.to_numeric(VentaPromedio,errors='coerce')
Faltante = pd.to_numeric(faltante,errors='coerce',downcast='integer')

numeric = pd.concat([Existencia,Desplazamiento,VentaPromedio],axis=1)


final_data = pd.concat([categorical,numeric,Faltante],axis=1)

## Vamos a probar descartando las columnas 'DEPTO' , 'SDEPTO' , 'CLASE'  y  'SCLASE'

data1 = final_data.drop(columns=['DEPTO','SDEPTO','CLASE','SCLASE'])

# Descartamos los datos nulos 
data1 = data1.dropna()

# Revisamos la proporción de datos
target = data1['Y_Faltante']
yes = target[target == 1].count()
no = target[target == 0].count()
print('Para el dataset de 8 semanas tenemos')
print('yes %: ' + str(yes/len(target)*100) + '- no %: ' \
        + str(no/len(target)*100))

#Balanceamos a 50% - 50% nuestro conjunto de datos.
data_yes = data1[data1["Y_Faltante"] == 1]
data_no = data1[data1["Y_Faltante"] == 0]
over_sampling = data_yes.sample(no, replace=True, \
random_state = 0)
balanced_data = pd.concat([data_no, over_sampling], \
axis=0)
data = balanced_data.reset_index(drop=True)

data = data.sample(frac=1).reset_index(drop=True)


categorical = data.drop(columns=['Existencia','fiDesplazamiento','VentaPromedio','Y_Faltante'])
numerical = data[['Existencia','fiDesplazamiento','VentaPromedio']]
numerical_scaled = (numerical - numerical.min())/(numerical.max() - numerical.min()) 
faltante = data["Y_Faltante"]


scaled_data = pd.concat([categorical,numerical_scaled,faltante], axis=1)

from fastai.data.transforms import RandomSplitter
splits = RandomSplitter(valid_pct=0.2)(range_of(data))

to = TabularPandas(data, procs=[Categorify,Normalize],
                   cat_names = ['Tienda','FORMATO','REGION','Categoria','Proveedor','Mes','DiaSem','Sem','Decil','Importado','MarcaPropia','EnOferta',],
                   cont_names = ['Existencia','fiDesplazamiento','VentaPromedio'],
                   y_names='Y_Faltante',
                   splits=splits)


dls = to.dataloaders(bs=64)

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(1)
learn.show_results()

# Construímos nuestros conjuntos para entenar y probar
X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()


################# Construimos un conjunto de validación #####################
# Abrir el archivo de prueba
file = 'Datos_faltante_1semana_20220606 (no balanceado para probar).csv'
data_test = pd.read_csv(file, header=0,low_memory=True)#, usecols=data_columns)
print('Aquí las dimensiones del conjunto no balanceado')
print('rows:', data_test.shape[0], ' columns:', data_test.shape[1])

# Aislamos las variables numéricas y realizamos un min-max scaling sobre ellas
categorical = data_test.drop(columns=['Existencia','fiDesplazamiento','VentaPromedio','Y_Faltante'])
numerical = data_test[['Existencia','fiDesplazamiento','VentaPromedio']]
#numerical_scaled = (numerical - numerical.min())/(numerical.max() - numerical.min()) 
faltante = data_test["Y_Faltante"]


Existencia = numerical['Existencia']
Desplazamiento = numerical['fiDesplazamiento']
VentaPromedio = numerical['VentaPromedio']

Existencia = pd.to_numeric(Existencia,errors='coerce')
Desplazamiento = pd.to_numeric(Desplazamiento,errors='coerce')
VentaPromedio = pd.to_numeric(VentaPromedio,errors='coerce')
Faltante = pd.to_numeric(faltante,errors='coerce',downcast='integer')

# Esta línea fué añadida.
numeric = pd.concat([Existencia,Desplazamiento,VentaPromedio],axis=1)


final_data = pd.concat([categorical,numeric,Faltante],axis=1)


## Vamos a probar descartando las columnas 'DEPTO' , 'SDEPTO' , 'CLASE'  y  'SCLASE'
data1 = final_data.drop(columns=['DEPTO','SDEPTO','CLASE','SCLASE'])

# Descartamos los datos nulos 
data1 = data1.dropna() ######## Con esto construiremos los conjuntos de validación.

#data = data1.sample(frac=1).reset_index(drop=True) ######### Esta estructura será usada para generar el reporte. 

categorical = data.drop(columns=['Existencia','fiDesplazamiento','VentaPromedio','Y_Faltante'])
numerical = data[['Existencia','fiDesplazamiento','VentaPromedio']]
numerical_scaled = (numerical - numerical.min())/(numerical.max() - numerical.min()) 
faltante = data["Y_Faltante"]

scaled_data = pd.concat([categorical,numerical_scaled,faltante], axis=1)

######################## Reintegrar ################################

val = TabularPandas(data1, procs=[Categorify,Normalize],
                   cat_names = ['Tienda','FORMATO','REGION','Categoria','Proveedor','Mes','DiaSem','Sem','Decil','Importado','MarcaPropia','EnOferta',],
                   cont_names = ['Existencia','fiDesplazamiento','VentaPromedio'],
                   y_names='Y_Faltante',
                   )

#dls = val.dataloaders(bs=64)
#learn = tabular_learner(dls, metrics=accuracy)
#learn.fit_one_cycle(1)
#learn.show_results()

######################## Esto podría posiblemente mejorarse  

X_val = val.xs 
y_val = val.ys.values
############################################



seed = 0
tf.random.set_seed(0)


def ago_classifier(activation,dropout_rate,optimizer,depth,epochs):
    model = Sequential()
    checkpoint_path = 'tf_tabular/cp.ckpt'
    checkpoint_dir=os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelChecpoint(filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1)
    callback = tf.keras.callbacks.EarlyStopping(monitor=['loss','val_accuracy'],patience=5)

    if(activation=='selu'):
        model.add(Dense(200,input_shape=(X_train.shape[1],),activation=activation,kernel_initializer='lecun_normal'))
        for n in range(depth-1):
            model.add(Dense(100,activation=activation,kernel_initializer='lecun_normal'))
            model.add(tf.keras.layers.AlphaDropout(rate=dropout_rate))
        model.add(Dense(20,activation=activation,kernel_initializer='lecun_normal'))
        model.add(Dense(1,activation='sigmoid'))
  
    else:
        model.add(Dense(200,input_shape=(X_train.shape[1],),activation=activation))
        for n in range(depth-1):
            model.add(Dense(100,activation=activation))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1,activation='sigmoid'))
    model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=optimizer,
            metrics=['accuracy',tf.keras.metrics.Recall()]
            )
    model.fit(X_train,y_train,
            epochs=epochs,
            validation_data=(X_test,y_test),
            callbacks=[callback,cp_callback])
    return model


lr = 0.01
optimizer = tf.keras.optimizers.Adam(lr=lr)
loss = tf.keras.losses.BinaryCrossentropy()

model = ago_classifier('leaky-relu',0.1,optimizer,10,100)

model.compile(optimizer=optimizer,
        loss=loss,
        metrics=['accuracy',tf.keras.metrics.Recall()])
model_history = model.fit(X_train,y_train,
        epochs=100,
        validation_data=(X_test,y_test))

loss_graph = pd.DataFrame(model_history.history).plot(title='History')
loss_graph
plt.show()


preds_proba = model.predict(X_val)
preds = preds_proba >= 0.5

acc = Accuracy()
prec = Precision()
rec = Recall()


acc.update_state(preds,y_val)
acc_results = acc.result().numpy()
print(f'Accuracy Score : {acc_results}')


prec.update_state(preds,y_val)
prec_results = prec.result().numpy()
print(f'Precision Score : {prec_results}')


rec.update_state(preds,y_val)
rec_results = rec.result().numpy()
print(f'Recall Score : {rec_results}')


f1 = 2*(prec_results * rec_results) / (prec_results + rec_results)
print(f'F1 Score {f1}')












