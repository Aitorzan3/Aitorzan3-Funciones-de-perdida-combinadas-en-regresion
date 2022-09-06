# Importación de recursos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random as python_random
import time

import numpy as np
import pandas as pd

import tensorflow 
from tensorflow import keras

import keras.losses
from keras.models import Model
from keras.layers import Input,Dense
from keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_uniform
from keras.regularizers import L2

from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

import scikeras
from scikeras.wrappers import KerasClassifier, BaseWrapper, KerasRegressor
from joblib import dump
from basewrapper import BaseWrapper

# Método para la reducción de la dimensionalidad de las coordenadas de Sotavento
def rejilla(dataset):
    coordenadas = ["(43.625, -8.125)","(43.625, -8.0)","(43.625, -7.875)","(43.625, -7.75)","(43.625, -7.625)",
               "(43.5, -8.125)","(43.5, -8.0)","(43.5, -7.875)","(43.5, -7.75)","(43.5, -7.625)",
               "(43.375, -8.125)","(43.375, -8.0)","(43.375, -7.875)","(43.375, -7.75)","(43.375, -7.625)",
               "(43.25, -8.125)","(43.25, -8.0)","(43.25, -7.875)","(43.25, -7.75)","(43.25, -7.625)",
               "(43.125, -8.125)","(43.125, -8.0)","(43.125, -7.875)","(43.125, -7.75)","(43.125, -7.625)"]
    variables = ["10u_", "10v_", "2t_", "sp_", "100u_", "100v_", "vel10_", "vel100_"]
    
    indices = []
    for var in variables:
        for coord in coordenadas:
            indices.append(str(var) + str(coord))
    return(dataset[indices])

# Definición de una función para cargar los datos del problema Sotavento
def sotavento_data():
    sota2016 = pd.read_csv("data/data_target_stv_2016.csv", sep = ',',  encoding = "ISO-8859-1", index_col=0, parse_dates=True).dropna()
    sota2017 = pd.read_csv("data/data_target_stv_2017.csv", sep = ',',  encoding = "ISO-8859-1", index_col=0, parse_dates=True).dropna()
    test = pd.read_csv("data/data_target_stv_2018.csv", sep = ',',  encoding = "ISO-8859-1", index_col=0, parse_dates=True).dropna()
    train = sota2016.append(sota2017)
    y_train = train[['targ']].to_numpy()
    x_train = rejilla(train.drop(columns = ['targ'])).to_numpy()
    y_test = test[['targ']].to_numpy()
    x_test = rejilla(test.drop(columns = ['targ'])).to_numpy()
    return(x_train,y_train,x_test,y_test)

# Definición de la función para establecer el determinismo de los experimentos junto con sus semillas
def set_seed(seed):

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    os.environ['PYTHONHASHSEED'] = str(seed)
    python_random.seed(seed)
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)

# Definición de la red neuronal artificial
def network(l2):

    set_seed(123)
    input_layer = Input(shape=(200,),name = "input_layer")
    layer_1 = Dense(200, activation="relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'layer_1')(input_layer)
    layer_2 = Dense(100, activation="relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'layer_2')(layer_1)
    output_layer = Dense(1, activation="linear", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'output_layer')(layer_2)
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
                                        epochs = self.epochs, initial_epoch = 0, shuffle = False,
                                       callbacks = [es])

        return self.history_

    def predict(self, x, **kwargs):
        pred = BaseWrapper.predict(self, x)
        return np.clip(pred, a_min = 0, a_max = 1)
    
    def score(self, x, y):
        y_pred = BaseWrapper.predict(self, x)
        y_pred = np.clip(y_pred, a_min = 0, a_max = 1)
        return mean_absolute_error(y, y_pred)
    
if __name__ == '__main__':
    set_seed(123)
    
    # Carga de los datos del problema Abalone
    x_train,y_train,x_test,y_test = sotavento_data()
    
    # Inicialización de las callbacks para el entrenamiento
    es = keras.callbacks.EarlyStopping(monitor="loss",
                                        min_delta=1.e-8,
                                        patience=50,
                                        verbose=0,
                                        mode="auto",
                                        baseline=None,
                                        restore_best_weights=True,
                                    )

    # Definición de las particiones para los datos de la validación cruzada
    n_folds = 2
    kfold = KFold(n_splits=n_folds, shuffle=False)
    
    # Definición del rango de los hiperparámetros para su ajuste mediante validación cruzada
    l_alpha = [10.**k for k in range(-6, 5)]

    param_grid = {'regressor__mlp__l2': l_alpha}
    
    # Definición del modelo
    regressor = CompanionWrapper(network,epochs=10, verbose=0, l2= 0.01, batch_size = 256, callbacks = [es], optimizer= 'adam')
    
    # Definición del pipeline y el escalador de los valores objetivo del modelo  
    regr = Pipeline(steps=[('std_sc', StandardScaler()),
                                      ('mlp', regressor)])
    
    y_transformer = MinMaxScaler()
    inner_estimator = TransformedTargetRegressor(regressor=regr,
                                                 transformer=y_transformer)

    # Definición de la función de puntuación para la validación cruzada
    mi_scoring = make_scorer(fc_mae) 
    
    # Búsqueda exhaustiva y ajuste de los hiperparámetros del modelo.
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
    _ = cv_estimator_0.fit(x_train, y_train)
    t_1 = time.time() 
    print("\nKerasClassifier_grid_search_time: %.2f" % ((t_1 - t_0)/60.))

    # Impresión de los resultados del entrenamiento y el ajuste de los hiperparámetros del modelo 
    l2 = cv_estimator_0.best_params_['regressor__mlp__l2']
    score =-cv_estimator_0.best_score_
    print("Keras Regressor \n")
    print("l2 = " + str(l2) + "\n" + "score = " + str(score))
    
    # Evaluación del modelo en el conjunto de test
    y_pred_mlp_cv = cross_val_predict(cv_estimator_0.best_estimator_, x_test, y_test, cv=kfold)
    err=y_test - y_pred_mlp_cv
    print("mae: %.3f" % (abs(err).mean()) )
    cvscore = cross_val_score(cv_estimator_0.best_estimator_, x_test, y_test, scoring =mi_scoring, cv=kfold)
    print("cross_val_score" + str(cvscore) )
    
    # Guardado del modelo óptimo para el posterior análisis de resultados
    dump(cv_estimator_0, 'regr_wrap.joblib')