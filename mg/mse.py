# Importación de recursos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random as python_random
from numpy.random import seed
from tensorflow.random import set_seed


import time
import warnings


import numpy as np
import pandas as pd

import tensorflow 
from tensorflow import keras

import keras.losses
from keras.models import Model
from keras.layers import Input,Dense
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_uniform
from keras.regularizers import L2


from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

import scikeras
from scikeras.wrappers import KerasClassifier, BaseWrapper, KerasRegressor
from basewrapper import BaseWrapper

from joblib import dump, load

# Definición de la red neuronal artificial
def network(l2=0.01):
    set_seed(123)
    input_layer = Input(shape=(6,), name = 'input_layer')
    layer_1 = Dense(50, activation="relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'layer_1')(input_layer)
    output_layer= Dense(1, activation="linear", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'output_layer')(layer_1)

    # Definción del modelo con las capas de entrada y salida
    model = Model(inputs=input_layer, outputs=output_layer)
    # Definición del optimiziador y la función de pérdida para cada salida
    model.compile(optimizer='adam', loss='mse', metrics = 'mae')
    return model

# Definición de una función de puntuación basada en el error absoluto medio
def fc_mae(y, y_pred, **kwargs):
    return -mean_absolute_error(y, y_pred) 

# Definición de la subclase de la clase BaseWrapper para implementar el modelo mse
class CompanionWrapper(BaseWrapper):
        
    def __init__(self, model = network, **sk_params):
        BaseWrapper.__init__(self, model=model, **sk_params)
    

    def fit(self, x, y, **kwargs):
    	  
        es = keras.callbacks.EarlyStopping(monitor="loss",
                                        min_delta=1.e-8,
                                        patience=50,
                                        verbose=0,
                                        mode="auto",
                                        baseline=None,
                                        restore_best_weights=True,
    
        )
	  
        y_regr = y
        
        self.X_dtype_ = type(x)
        self.X_shape_ = x.shape


        self.history_ = BaseWrapper.fit(self, x, {'output_layer':y_regr},
                                        epochs = self.epochs, initial_epoch = 0, callbacks = [es])

        return self.history_

    def predict(self, x, **kwargs):
        pred = BaseWrapper.predict(self, x)
        return np.clip(pred, a_min = 0, a_max = 1)
    
    def score(self, x, y):
        y_pred = BaseWrapper.predict(self, x)
        y_pred = np.clip(y_pred, a_min = 0, a_max = 1)
        return mean_absolute_error(y, y_pred)
    
# Definición de una función para cargar los datos del problema mg
def mg_data():
    data = pd.read_csv("datos.csv", sep = ' ',  header=None, encoding = "ISO-8859-1").dropna()
    data[1] = data[1].replace({'1:':''}, regex=True).astype(float)
    data[2] = data[2].replace({'2:':''}, regex=True).astype(float)
    data[3] = data[3].replace({'3:':''}, regex=True).astype(float)
    data[4] = data[4].replace({'4:':''}, regex=True).astype(float)
    data[5] = data[5].replace({'5:':''}, regex=True).astype(float)
    data[6] = data[6].replace({'6:':''}, regex=True).astype(float)
    x = data[[1,2,3,4,5,6]].values
    y = data[0].values
    return x,y

# Definición de la función para establecer el determinismo de los experimentos junto con sus semillas
def set_seed(seed):

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    os.environ['PYTHONHASHSEED'] = str(seed)
    python_random.seed(seed)
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    set_seed(123)
    
    # Carga de los datos del problema
    x,y = mg_data()
    
    # Inicialización de las callbacks para el entrenamiento    
    es = tensorflow.keras.callbacks.EarlyStopping(monitor="loss",
                                        min_delta=1.e-8,
                                        patience=50,
                                        verbose=0,
                                        mode="auto",
                                        baseline=None,
                                        restore_best_weights=True,
                                    )
    

    # Definición de las particiones para los datos de la validación cruzada
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True)
    
    # Definición del rango de los hiperparámetros para su ajuste mediante validación cruzada
    l_alpha = [10.**k for k in range(-6, 5)]
    param_grid = {'regressor__mlp__l2': l_alpha}

    # Definición del modelo
    regressor = CompanionWrapper(network, epochs=10000, verbose=0, l2= 0.01, batch_size = 256, optimizer = 'adam', callbacks = [es])
 
    # Definición del pipeline y el escalador de los valores objetivo del modelo
    regr = Pipeline(steps=[('std_sc', StandardScaler()), 
                           ('mlp', regressor)])

    y_transformer = MinMaxScaler()
    inner_estimator = TransformedTargetRegressor(regressor=regr,
                                                 transformer=y_transformer)

    # Definición de la función de puntuación para la validación cruzada
    mi_scoring = make_scorer(fc_mae)
    
    # Entrenamiento y ajuste de los hiperparámetros del modelo
    cv_estimator_0 = GridSearchCV(inner_estimator, 
                                  param_grid=param_grid, 
                                  cv=kfold, 
                                  scoring=mi_scoring, 
                                  return_train_score=True,
                                  refit=True,
                                  n_jobs = 5,
                                  verbose=1)

    t_0 = time.time()
    set_seed(123)
    _ = cv_estimator_0.fit(x, y)
    t_1 = time.time() 
    print("\nKerasClassifier_grid_search_time: %.2f" % ((t_1 - t_0)/60.))

    # Impresión de los resultados del entrenamiento y el ajuste de los hiperparámetros del modelo   
    l2 = cv_estimator_0.best_params_['regressor__mlp__l2']
    score =-cv_estimator_0.best_score_
    print("Keras Regressor \n")
    print("l2 = " + str(l2) + "\n" + "score = " + str(score))

    # Guardado del modelo óptimo para el posterior análisis de resultados
    dump(cv_estimator_0, 'regr_wrap.joblib')