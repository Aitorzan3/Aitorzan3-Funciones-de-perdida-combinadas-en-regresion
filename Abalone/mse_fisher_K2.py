# Importación de recursos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import random as python_random
from numpy.random import seed
from tensorflow.random import set_seed

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
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import scikeras
from basewrapper import BaseWrapper
from joblib import dump

# Definición de la red neuronal artificial
def network(l2, lw):
    set_seed(123)
    input_layer = Input(shape=(8,),name = "input_layer")
    layer_1 = Dense(50, activation="relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'layer_1')(input_layer)
    layer_2 = Dense(50, activation="relu", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'layer_2')(layer_1)
    regr_layer = Dense(1, activation="linear", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'regr_layer')(layer_2)
    clas_layer = Dense(2, activation="softmax", kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l2(l2), name = 'clas_layer')(layer_2)
    # Definción del modelo con las capas de entrada y salida
    network_ = Model(inputs=[input_layer], outputs=[regr_layer, clas_layer])
    # Definición del optimiziador y la función de pérdida para cada salida
    network_.compile(optimizer='adam',
                   loss={'regr_layer':'mse','clas_layer': 'mse'},
                   loss_weights= {"regr_layer":1.0, "clas_layer": lw},
                   metrics = {'regr_layer':'mae', "clas_layer": 'accuracy'})
    return(network_)

# Definición de la función para establecer el determinismo de los experimentos junto con sus semillas
def set_seed(seed):

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    os.environ['PYTHONHASHSEED'] = str(seed)
    python_random.seed(seed)
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)

# Definición de un escalador parcial para los modelos con varias salidas basado en MinMaxScaler
class PartialMinMaxScaler(MinMaxScaler):
    """Applies the scaler in super() to the first column of the 
    numpy array to be scaled,
    """
    def fit(self, x):
        super().fit(x[ : , 0].reshape(-1, 1))
        
    def transform(self, x):
        xx = x.copy()
        xx_0 = super().transform(x[ : , 0].reshape(-1, 1))
        xx = np.concatenate([xx_0.reshape(-1,1),xx[:,1].reshape(-1,1)], axis = 1)
        return xx
        
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        xx = x.copy()
        x_0 = super().inverse_transform(x[ : , 0].reshape(-1, 1))
        xx[ : , 0] = x_0.reshape(-1, )
        return xx

# Definición de una función de puntuación para los modelos con múltiples salidas basada en el error absoluto medio
def fc_mae(y, y_pred, **kwargs):
    return -mean_absolute_error(y[ : , 0], y_pred) 

# Definición de una función para cargar los datos del problema Abalone
def abalone_data():
    data = pd.read_csv("datos.csv", sep = ' ',  header=None, encoding = "ISO-8859-1").dropna()
    data[1] = data[1].replace({'1:':''}, regex=True).astype(int)
    data[2] = data[2].replace({'2:':''}, regex=True).astype(float)
    data[3] = data[3].replace({'3:':''}, regex=True).astype(float)
    data[4] = data[4].replace({'4:':''}, regex=True).astype(float)
    data[5] = data[5].replace({'5:':''}, regex=True).astype(float)
    data[6] = data[6].replace({'6:':''}, regex=True).astype(float)
    data[7] = data[7].replace({'7:':''}, regex=True).astype(float)
    data[8] = data[8].replace({'8:':''}, regex=True).astype(float)
    x = data[[1,2,3,4,5,6,7,8]].values
    y = data[0].values
    return x,y

# Definición de la función para obtener la matriz de resultados para la pérdida de Fisher
def fisher_y(y):
    """ y_ij = (n-n_j)/(n*sqrt(n_j)) if class(y_i)==j else -sqrt(n_j)/n """
    n = len(y)
    targets, counts = np.unique(y, return_counts=True)
    c = len(targets)
    lsrlda_y = np.zeros((n, c))
    for i, target in enumerate(y):
        t = targets.tolist().index(target)
        for j in range(c):
            if j==t:
                lsrlda_y[i, j] = (n - counts[t]) / (n * np.sqrt(counts[t]))
            else:
                lsrlda_y[i, j] = -np.sqrt(counts[t]/n)
    return lsrlda_y

# Definición de la subclase de la clase BaseWrapper para implementar el modelo mse + Fisher
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
        
        # Tratamiento de los valores objetivo de las instancias para cada salida del modelo
        y_regr = y[:,0]
        y_clas = y[:,1]
        fishery = fisher_y(y_clas)
        
        self.X_dtype_ = type(x)
        self.X_shape_ = x.shape
        
        self.history_ = BaseWrapper.fit(self, x,
                                        {'regr_layer':y_regr,'clas_layer':fishery},
                                        epochs = self.epochs, initial_epoch = 0,
                                       callbacks = [es])

        return self.history_


    def predict(self, x, **kwargs):
        """
        Utilizamos únicamente las predicciones de la salida lineal
        """
        pred = BaseWrapper.predict(self, x)[0]
        return np.clip(pred, a_min = 0, a_max = 1)
    
    def score(self, x, y):
        y_pred = BaseWrapper.predict(self, x)[0]
        y_pred = np.clip(y_pred, a_min = 0, a_max = 1)
        return mean_absolute_error(y[:,0], y_pred)


if __name__ == '__main__':
    set_seed(123)
    
    # Carga de los datos del problema Abalone
    x,y = abalone_data()
    
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
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True)
    
    # Definición del rango de los hiperparámetros para su ajuste mediante validación cruzada
    l_alpha = [10.**k for k in range(-6, 5)]
    lw_values = [0.05*k for k in range(0, 20)]
    param_grid = {'regressor__mlp__lw': lw_values,
                  'regressor__mlp__l2': l_alpha}
    
    # Definición del modelo mse + Fisher
    companion = CompanionWrapper(network,
                    batch_size=256,
                    epochs=10000,
                    verbose=0,
                    l2 = 0.01,
                    lw = 0.5,
                    optimizer = 'adam',
                    callbacks = [es]
                    )

    # Definición del pipeline y el escalador de los valores objetivo del modelo
    regr = Pipeline(steps=[('std_sc', StandardScaler()),
                       ('mlp', companion)]) 
    y_transformer = PartialMinMaxScaler()
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
                                  refit = True, 
                                  n_jobs = 5,
                                  verbose=0)
    
    
    # Tratamiento de los valores objetivo para las salidas del modelo
    q = np.quantile(y, [0,.5,1])
    y_clas = np.array(pd.cut(x=y.reshape(-1,), bins=q, labels=[0, 1], include_lowest=True, right=True))
    target = np.concatenate([y.reshape(-1,1),y_clas.reshape(-1,1)], axis = 1)
    t_0 = time.time() 
    set_seed(123)
    _ = cv_estimator_0.fit(x, target)
    t_1 = time.time() 
    print("\nComp_grid_search_time: %.2f" % ((t_1 - t_0)/60.))

    # Impresión de los resultados del entrenamiento y el ajuste de los hiperparámetros del modelo   
    lw = cv_estimator_0.best_params_['regressor__mlp__lw']
    l2 = cv_estimator_0.best_params_['regressor__mlp__l2']
    score =-cv_estimator_0.best_score_
    print("Companion Loss mse - Fisher K=2")
    print("l2 = " + str(l2) + "\n lw " + str(lw) + "\n" + "score = " + str(score))

    # Guardado del modelo óptimo para el posterior análisis de resultados
    dump(cv_estimator_0, 'comp_wrap2_K2.joblib')
    
    