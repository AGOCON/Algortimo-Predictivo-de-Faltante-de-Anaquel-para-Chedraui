####### Este script genera un reporte para Formato completo. 

# Hay que analizar el tema de la normalización antes de encajar los datos categoricos.
# This is the first implementation
import math
import numpy as np
import cuml as sklearn
#import pandas as pd
#import keras_tuner
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import seaborn as sns
from fastai.tabular.all import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import cudf as pd


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

numeric = pd.concat([Existencia,Desplazamiento,VentaPromedio],axis=1, ignore_index=True)


#print(Faltante.columns)

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


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=seed)
rfc.fit(X_train, y_train)
rfc.predict(X_test)
acc_train = rfc.score(X_train,y_train)
acc_test = rfc.score(X_test,y_test)
print(f'Precisión entrenamiento: {acc_train.round(4)}, Presición validación: {acc_test.round(4)}')




from sklearn.metrics import confusion_matrix
#y_test = label_test     ########### Esto NO debe ser uncommented. 
y_pred = rfc.predict(X_val)
mat = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 9))
sns.heatmap(mat, annot=True)
plt.show()
plt.savefig('Random_Forest_Confusion_Matrix.png')

accuracy = accuracy_score(y_val,y_pred)
recall = recall_score(y_val,y_pred)
precision = precision_score(y_val,y_pred)
balanced_accuracy = balanced_accuracy_score(y_val,y_pred)
f1 = f1_score(y_val,y_pred)
print(f'Accuracy: {accuracy}')
print(f'True Positive Rate {recall}')
print(f'Precision {precision}')
print(f'Balanced accuracy {balanced_accuracy}')
print(f'f1 score {f1}')





################ Esta función requiere como input 'data1' para producir resultados.
def reporte_formato(frame):
    l = len(frame['FORMATO'].value_counts())
    formatos = frame['FORMATO'].value_counts()
    def segmentador_formato(frame,cadena):
        arreglo = frame[frame['FORMATO']==cadena]
        return arreglo
    lineas = []
    for i in range(l):
        cadena = formatos.iloc[i:i+1].index[0]
        formato = segmentador_formato(frame,cadena)
        encaje = TabularPandas(formato, procs=[Categorify,Normalize],
                   cat_names = ['Tienda','FORMATO','REGION','Categoria','Proveedor','Mes','DiaSem','Sem','Decil','Importado','MarcaPropia','EnOferta',],
                   cont_names = ['Existencia','fiDesplazamiento','VentaPromedio'],
                   y_names='Y_Faltante',
                   )
        objective = encaje.xs
        target = encaje.ys.values
        y_pred = rfc.predict(objective)
        accuracy = accuracy_score(target,y_pred)
        recall = recall_score(target,y_pred)
        precision = precision_score(target,y_pred)
        balanced_accuracy = balanced_accuracy_score(target,y_pred)
        f1 = f1_score(target,y_pred)
        lineas.append(f'#####Calificaciones obtenidas para el formato {i} ########## \n')
        lineas.append(f'\n')
        lineas.append(f'\n')
        lineas.append(f'Nombre de formato :  ' + cadena)
        lineas.append(f'Accuracy: {accuracy} \n')
        lineas.append(f'True Positive Rate: {recall} \n')
        lineas.append(f'Precision: {precision} \n')
        lineas.append(f'Balanced accuracy: {balanced_accuracy} \n')
        lineas.append(f'f1 score: {f1} \n')
        lineas.append(f'\n')
        lineas.append(f'\n')
    return lineas


########################### Reporte por categorias

def reporte_categorias(frame):
    l = len(frame['Categoria'].value_counts())
    categorias = frame['Categoria'].value_counts()
    def segmentador_categoria(frame,cadena):
        arreglo = frame[frame['Categoria']==cadena]
        return arreglo
    lineas = []
    dicionario_f1 = {}
    dicionario_recall = {}
    for i in range(l):
        cadena = categorias.iloc[i:i+1].index[0]
        categoria = segmentador_categoria(frame,cadena)
        encaje = TabularPandas(categoria, procs=[Categorify,Normalize],
                   cat_names = ['Tienda','FORMATO','REGION','Categoria','Proveedor','Mes','DiaSem','Sem','Decil','Importado','MarcaPropia','EnOferta',],
                   cont_names = ['Existencia','fiDesplazamiento','VentaPromedio'],
                   y_names='Y_Faltante',
                   )
        objective = encaje.xs
        target = encaje.ys.values
        y_pred = rfc.predict(objective)
        accuracy = accuracy_score(target,y_pred)
        recall = recall_score(target,y_pred)
        precision = precision_score(target,y_pred)
        balanced_accuracy = balanced_accuracy_score(target,y_pred)
        f1 = f1_score(target,y_pred)
        lineas.append(f'#####Calificaciones obtenidas para el formato {i} ########## \n')
        lineas.append(f'\n')
        lineas.append(f'\n')
        lineas.append(f'Categoria  :  ' + cadena)
        lineas.append(f'Accuracy: {accuracy} \n')
        lineas.append(f'True Positive Rate: {recall} \n')
        lineas.append(f'Precision: {precision} \n')
        lineas.append(f'Balanced accuracy: {balanced_accuracy} \n')
        lineas.append(f'f1 score: {f1} \n')
        lineas.append(f'\n')
        lineas.append(f'\n')
        dicionario_f1[cadena] = f1
        dicionario_recall[cadena] = recall
    return [lineas,dicionario_recall,dicionario_f1]


# Función que ordena un diccionario por los valores de sus valores (dictionary values).
def ordena_dic(dic=dict):
    new_dic = dict(sorted(dic.items(),key= lambda x:x[1]))
    return new_dic


################### Estructuras de datos que almacenan nuestros resultados para reporte a desempeño por categorias


texto_categoria = reporte_categorias(data1)[0] # Esto es una lista
reporte_recall = ordena_dic(reporte_categorias(data1)[1]) # Esto es un diccionario 
reporte_f1 = ordena_dic(reporte_categorias(data1)[2]) # Esto también es un diccionario 


########## Esta línea produce el reporte de formato sobre el conjunto de validación
reporte_f = reporte_formato(data1)
with open('Reporte_Formato.txt','w') as f:
    f.write('\n'.join(reporte_f))

####################################################################3
################################# Este bloque produce un reporte para desempeño por categorias sobre nuestro conjunto de validación #####################


with open('Reporte_Formato.txt','w') as f:
    f.write('\n'.join(reporte_f))


# Imprimimos un reporte de clasificación
from sklearn.metrics import classification_report
target_names = ['Faltante', 'No Faltante']
print(classification_report(y_val, y_pred, target_names=target_names))


 
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

with open('Reporte_General_Voting.txt','w') as f:
    f.write(f'#####Calificaciones obtenidas por el algoritmo de votación ########## \n')
    f.write(f'Accuracy: {accuracy} \n')
    f.write(f'True Positive Rate: {recall} \n')
    f.write(f'Precision: {precision} \n')
    f.write(f'Balanced accuracy: {balanced_accuracy} \n')
    f.write(f'f1 score: {f1} \n')
    f.write(f'\n')
    f.write(f'tn: {tn}')
    f.write(f'fp: {fp}')
    f.write(f'fn: {fn}')
    f.write(f'tp: {tp}')


from sklearn.metrics import roc_curve
y_pred_proba = rfc.predict_proba(X_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
plt.plot([0,1],[0,1],'k-')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve for the voting Classifier')
plt.show()
plt.savefig('Voting_ROC_curve.png')


reporte_recall 
reporte_f1 

with open('Reporte_recall_por_categoria.txt','w') as g:
    r1 = list(reporte_recall.keys())
    r2 = list(reporte_recall.values())
    g.write(f'Calificaciones recall por categoría')
    for i in range(len(r1)):
        g.write('\n')
        g.write(f'{r1[i]}' + f':  ')
        g.write(f'{r2[i]}')
        g.write('\n')


with open('Reporte_f1_por_categoria.txt','w') as h:
    f1 = list(reporte_recall.keys())
    f2 = list(reporte_recall.values())
    h.write(f'Calificaciones recall por categoría')
    for i in range(len(r1)):
        h.write('\n')
        h.write(f'{r1[i]}'+f':  ')
        h.write(f'{r2[i]}')
        h.write('\n')


