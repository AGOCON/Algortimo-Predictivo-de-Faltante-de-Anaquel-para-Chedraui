# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# !pip install scikit-learn
# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension
# !pip install seaborn
# %%
# Importar librerías básicas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bctools as bc
# %%
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
# %%
# Especificar columnas
relevant_columns = ['SKU','Tienda','FORMATO','Categoria','CLASE','SCLASE','Sem','Existencia','fiDesplazamiento','VentaPromedio','Y_Faltante']
num_columns = ['Existencia','fiDesplazamiento','VentaPromedio']
cat_columns = ['SKU','Tienda','FORMATO','Categoria','CLASE','SCLASE','Sem']
target = ['Y_Faltante']
# %%
dtype = {'SKU':'object','FORMATO':'object','REGION':'object','Tienda':'object','Categoria':'object','Proveedor':'object','CLASE':'object','SCLASE':'object','Mes':'object','DiaSem':'object','Sem':'object','Existencia':'float','fiDesplazamiento':'float',
\
'VentaPromedio':'float','Y_Faltante':'int'}
# %%
# Abrir el archivo
file = '1semana.csv'
data = pd.read_csv(file, header=0,low_memory=False, usecols=relevant_columns,dtype=dtype)
print('rows:', data.shape[0], ' columns:', data.shape[1])
# %%
data = data.dropna()
# %%
# Verificamos que el conjunto de datos está perfectamente balanceado
# Revisamos la proporción de datos
target = data['Y_Faltante']
yes = target[target == 1].count()
no = target[target == 0].count()
print('yes %: ' + str(yes/len(target)*100) + '- no %: ' \
        + str(no/len(target)*100))

fig, ax = plt.subplots(figsize=(10,5))
plt.bar("Faltante", yes)
plt.bar("No Faltante", no)
ax.set_yticks([yes,no])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# %%
#Balanceamos a 50% - 50% nuestro conjunto de datos.
data_yes = data[data["Y_Faltante"] == 1]
data_no = data[data["Y_Faltante"] == 0]
over_sampling = data_yes.sample(no, replace=True, \
random_state = 0)
balanced_data = pd.concat([data_no, over_sampling], \
axis=0)
data = balanced_data.reset_index(drop=True)
# %%
# Revisamos (de nuevo) la proporción de datos
target = data['Y_Faltante']
yes = target[target == 1].count()
no = target[target == 0].count()
print('yes %: ' + str(yes/len(target)*100) + '- no %: ' \
        + str(no/len(target)*100))

fig, ax = plt.subplots(figsize=(10,5))
plt.bar("Faltante", yes)
plt.bar("No Faltante", no)
ax.set_yticks([yes,no])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# %%
X = data.drop('Y_Faltante',axis=1)
y = data.Y_Faltante
# %%
# Se construyen los conjuntos de entrenamiento y prueba sobre los que se aplicarán algoritmos de machine learning
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=0)
print('Entrenamiento:', X_train.shape[0], ' Prueba:', X_val.shape[0])
# %%
# Aislamos las variables numéricas y realizamos un min-max scaling sobre ellas
categorical = X_train.drop(columns=['Existencia','fiDesplazamiento','VentaPromedio'])
numerical = X_train[['Existencia','fiDesplazamiento','VentaPromedio']]
numerical_scaled = (numerical - numerical.min())/(numerical.max() - numerical.min())
X_train = pd.concat([categorical,numerical_scaled], axis=1)
# %%
indices_categoricos = np.where(X_train.dtypes != float)[0]
indices_categoricos
# %%
num_ind = np.where(X_train.dtypes != object)[0]
num_ind
# %%
#params = {
#    'l2_leaf_reg':int(2.0),
#    'custom_loss':[metrics.Accuracy()],
#    'random_seed': 0,
#    'task_type':'GPU',
#    'logging_level': 'Verbose',
#    'use_best_model': True,
#}
#train_pool = X_train, y_train
#validate_pool = (X_val, y_val)
# %%
#params_with_snapshot = params.copy()

#model = CatBoostClassifier(**params_with_snapshot)
#model.fit(X_train,y_train,cat_features=indices_categoricos, eval_set=(X_val,y_val),logging_level='Verbose',save_snapshot=True,plot=True)

#print('Simple model tree count: {}'.format(model.tree_count_))
#print('Simple model validation accuracy: {:.4}'.format(
#    accuracy_score(y_val, model.predict(X_val))
#))
#print('')
# %%
#model.save_model('Ago_model_medio.dump')
model = CatBoostClassifier()
model.load_model('Ago_model_medio.dump');
# %%
y_pred = model.predict(X_val)
# Dibujamos también la matriz de confusión para nuestro modelo
mat = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 9))
sns.heatmap(mat, annot=True)
plt.show()
# %%
# Calculamos las entradas de la matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
tn,fp,fn,tp
# %%
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val,y_pred)
recall = recall_score(y_val,y_pred,average='binary',pos_label=1)
precision = precision_score(y_val,y_pred,average='binary',pos_label=0)
balanced_accuracy = balanced_accuracy_score(y_val,y_pred)
f1 = f1_score(y_val,y_pred,pos_label=1)
print(f'Accuracy: {accuracy}')
print(f'True Positive Rate {recall}')
print(f'Precision {precision}')
print(f'Balanced accuracy {balanced_accuracy}')
print(f'f1 score {f1}')
# %%
# Dibujamos la curva de ROC para este modelo
from sklearn.metrics import roc_curve
y_pred_proba = model.predict_proba(X_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba,pos_label=1)
plt.plot([0,1],[0,1],'k-')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()
# %%
pip install binclass-tools
# %%
import bctools as bc
# %%
# Get prediction probabilities for the train set
train_predicted_proba = model.predict_proba(X_train)[:,1]

# Get prediction probabilities for the test set
test_predicted_proba = model.predict_proba(X_val)[:,1] 
# %%
area_under_ROC = bc.curve_ROC_plot(true_y= y_val, 
                                   predicted_proba = test_predicted_proba)
# %%
area_under_ROC
# %%
area_under_PR = bc.curve_PR_plot(true_y= y_val, 
                                 predicted_proba = test_predicted_proba, 
                                 beta = 1)
# %%
area_under_PR
# %%
threshold_step = 0.05

bc.predicted_proba_violin_plot(true_y = y_val, 
                               predicted_proba = test_predicted_proba, 
                               threshold_step = threshold_step)
# %%
# set params for the train dataset
threshold_step = 0.05
amounts = np.abs(X_train[:, 13])
optimize_threshold = 'all'
currency = '$'

# The function get_cost_dict can be used to define the dictionary of costs.
# It takes as input, for each class, a float or a list of floats. 
# Lists must have coherent lenghts 

train_cost_dict = bc.get_cost_dict(TN = 0, FP = 10, FN = np.abs(X_train[:, 12]), TP = 0)
# %%
var_metrics_df, invar_metrics_df, opt_thresh_df = bc.confusion_matrix_plot(
    true_y = y_train, 
    predicted_proba = train_predicted_proba, 
    threshold_step = threshold_step, 
    amounts = amounts, 
    cost_dict = train_cost_dict, 
    optimize_threshold = optimize_threshold, 
    #N_subsets = 70, subsets_size = 0.2, # default
    #with_replacement = False,           # default
    currency = currency,
    random_state = 123,
    title = 'Interactive Confusion Matrix for the Training Set');
# %%
# You can also analyze the test dataset.
# In this case there is no need to optimize the threshold value for any measure.
threshold_step = 0.05
amounts = np.abs(X_val[:, 13])
optimize_threshold = None
#currency = '$'

test_cost_dict = bc.get_cost_dict(TN = 0, FP = 10, FN = np.abs(X_val[:, 12]), TP = 0)

var_metrics_df, invar_metrics_df, __ = bc.confusion_matrix_plot(
    true_y = y_val, 
    predicted_proba = test_predicted_proba, 
    threshold_step = threshold_step, 
    amounts = amounts, 
    cost_dict = test_cost_dict, 
    optimize_threshold = optimize_threshold, 
    #N_subsets = 70, subsets_size = 0.2, # default
    #with_replacement = False,           # default
    #currency = currency,
    random_state = 123);
# %%
invar_metrics_df = bc.utilities.get_invariant_metrics_df(true_y = y_val, 
                                      predicted_proba = test_predicted_proba)
# %%
conf_matrix, metrics_fixed_thresh_df = bc.utilities.get_confusion_matrix_and_metrics_df(
    true_y = y_val, 
    predicted_proba = test_predicted_proba,
    threshold = 0.3 # default = 0.5
)
# %%

threshold_values = np.arange(0.05, 1, 0.05)

opt_thresh_df = bc.thresholds.get_optimized_thresholds_df(
    optimize_threshold = ['Kappa', 'Fscore', 'Cost'], 
    threshold_values = threshold_values, 
    true_y = y_train, 
    predicted_proba = train_predicted_proba,
    cost_dict = train_cost_dict, 
    
    # GHOST parameters (these values are also the default ones)
    N_subsets = 70,
    subsets_size = 0.2,
    with_replacement = False,
    
    random_state = 120)
# %%
opt_roc_threshold_value = bc.thresholds.get_optimal_threshold(
    y_train, 
    train_predicted_proba, 
    threshold_values,
    ThOpt_metrics = 'ROC', # default = 'Kappa'

    # GHOST parameters (these values are also the default ones) 
    N_subsets = 70,
    subsets_size = 0.2,
    with_replacement = False,

    random_seed = 120)
# %%
opt_cost_threshold_value = bc.thresholds.get_cost_optimal_threshold(
    y_train, 
    train_predicted_proba, 
    cost_dict = train_cost_dict,

    # GHOST parameters (these values are also the default ones) 
    N_subsets = 70,
    subsets_size = 0.2,
    with_replacement = False,

    random_seed = 120)
# %%
amount_cost_df, total_amount = bc.confusion_linechart_plot(
    true_y = y_val, 
    predicted_proba = test_predicted_proba, 
    threshold_step =  threshold_step, 
    amounts = amounts, 
    cost_dict = test_cost_dict, 
    currency = currency);
# %%
# this function requires a list of thresholds, instead of the step, for example:
threshold_values = np.arange(0, 1, 0.05)

# example without amounts
costs_df = bc.utilities.get_amount_cost_df(
    true_y = y_test, 
    predicted_proba = test_predicted_proba,
    threshold_values = threshold_values, 
    #amounts = amounts,  
    cost_dict = test_cost_dict)
# %%
amount_classes = ['TP', 'FP'] 
cost_classes = 'all'

total_cost_amount_df = bc.total_amount_cost_plot(
    true_y = y_val, 
    predicted_proba = test_predicted_proba, 
    threshold_step = threshold_step,
    amounts = amounts, 
    cost_dict = test_cost_dict,
    amount_classes = amount_classes,
    cost_classes = cost_classes,
    currency = currency);
# %%
# this function requires a list of thresholds, instead of the step, for example:
threshold_values = np.arange(0, 1, 0.05)

# example without amounts
costs_df = bc.utilities.get_amount_cost_df(
    true_y = y_val, 
    predicted_proba = test_predicted_proba,
    threshold_values = threshold_values, 
    #amounts = amounts,  
    cost_dict = test_cost_dict)
# %%
# for example, if we want the True Positive data points with a 0.7 threshold:
confusion_category = 'TP'

bc.get_confusion_category_observations_df(
    confusion_category = confusion_category, 
    X_data = X_val, 
    true_y = y_val, 
    predicted_proba = test_predicted_proba, 
    threshold = 0.7 # default = 0.5
)
# %%








