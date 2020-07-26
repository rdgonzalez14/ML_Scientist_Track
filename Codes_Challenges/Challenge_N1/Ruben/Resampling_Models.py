# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA OBTENCIÓN DE MODELOS CON RESAMPLING DEL CHALLENGE N°1
------------------------------------------------------------------------------------------------------------------------
Fecha: 14 de Junio de 2020
Consideraciones:
    - El código está diseñado para correr tanto en Windows como en Linux
    - Se standarizan features numéricos con escalas de magnitud diferentes
    - Se eliminan outliers extremos identificados en el EDA
    - Los features categóricos se codifican con OneHotEncoding
    - El resampling se realizará de las siguiente formas:
        a. Undersampling de la clase mayoritaria para quedar en ratio 1:1
        b. Oversampling con SMOTE de la minoritaria hasta un punto intermedio 1:1 (Ej: 2000:2000)
    - El modelo con resampling es un xgboost tuneado
    - El tuneo de hiperparámetros se realiza con BayesianOptimization (skopt)
        a.  BayesSearchCV de SKOpt tiende a entrenar 1 modelo por core
    - La métrica de desempeño es el score F1
Autor: Ruben D. Gonzalez R.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importando librerías y/o funciones necesarias
#-----------------------------------------------------------------------------------------------------------------------
import time
import platform
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray
from sklearn.preprocessing import scale, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from Functions.General_Functions import *

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
start_time=time.time()
OS=platform.system()
IsCloud = 0
Sampling_Method = 'Hybrid'
if Sampling_Method == 'Undersampling':
    Model_Name = 'XGBoost_Undersampled'
elif Sampling_Method == 'SMOTE':
    Model_Name = 'XGBoost_SMOTE'
else:
    Model_Name = 'XGBoost_UnderSMOTE'
Tune_Params = {'n_iter':[40, 250], 'n_points':[1, 5], 'n_jobs':[4, 20]}

# ----------------------------------------------------------------------------------------------------------------------
# Definición de funciones
#-----------------------------------------------------------------------------------------------------------------------
# ---------------------------------- Función para realizar resampling del dataset --------------------------------------
def Resample_Data(X, Y, Sampling_Method):
    # Realizando resampling del dataset según método especificado
    if Sampling_Method == 'Undersampling':
        X_res, Y_res = RandomUnderSampler(sampling_strategy=1.0, replacement=False).fit_resample(X, Y)
    elif Sampling_Method == 'SMOTE':
        X_res, Y_res = SMOTENC(sampling_strategy=1.0, k_neighbors=3, categorical_features=[28, 29]).fit_resample(X, Y)
    else:
        X_unsamp, Y_unsamp = RandomUnderSampler(sampling_strategy=0.0354, replacement=False).fit_resample(X, Y)
        X_res, Y_res = SMOTENC(sampling_strategy=0.2, k_neighbors=3, categorical_features= [28,29]).fit_resample(X_unsamp, Y_unsamp)
    return (X_res,Y_res)

# --------------------------Función para definir pipelines para el entrenamiento del modelo-----------------------------
def Train_Model(clf, params, X_train,Y_train, index):
    # Definiendo pipeline para features categóricos nominales
    categ_nom_transformer = Pipeline(steps=[('OneHot_encoding', OneHotEncoder(drop='first'))])
    # Definiendo el transformador de columnas para aplicar los pipelines de transformacion de features
    cat_nom_features = ['Card_Franchise', 'Status']
    preprocessor = ColumnTransformer(transformers=[('cat_nom', categ_nom_transformer, cat_nom_features)],
                                     remainder='passthrough')
    # Definiendo pipeline y ejecutando tuneo de hiperparámetros
    ML_Pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    Best_Model = BayesSearchCV(ML_Pipe, params, scoring='f1', cv=4, n_iter=Tune_Params['n_iter'][index],
                               n_jobs=Tune_Params['n_jobs'][index], n_points=Tune_Params['n_points'][index],
                               verbose=1).fit(X_train, Y_train)
    return (Best_Model.best_estimator_, Best_Model.best_params_, Best_Model.cv_results_)

# ----------------------------------------------------------------------------------------------------------------------
# Función con el cuerpo principal del "main"
#-----------------------------------------------------------------------------------------------------------------------
def main():
    # -------------------------------- Levantando datasets desde los archivos ".csv" -----------------------------------
    Data, Results = Load_Dataset(OS)

    # --------------------------------------- Manipulando y preparando dataset -----------------------------------------
    # Estandarizando variables con magnitudes elevadas y eliminando las originales
    Data["Time_scaled"] = scale(Data["Time"], with_mean=True, with_std=True)
    Data["Amount_scaled"] = scale(Data["Amount"], with_mean=True, with_std=True)
    Data["Card_Limit_scaled"] = scale(Data["Card_Limit"], with_mean=True, with_std=True)
    Data.drop(columns=['Time','Amount','Card_Limit'], inplace=True)
    # Eliminando outliers extremos detectados en el EDA
    Data = Drop_Outliers(Data)
    # Dividiendo dataset en train y test
    Label = Data['Class']
    Data.drop(columns='Class', inplace=True)
    X_train, X_test, Y_train, Y_test = train_test_split(Data, Label, test_size=0.2, stratify=Label)
    # Realizando resampling del dataset
    Init_Rows, Init_Pos = len(Y_train), sum(Y_train)
    X_train, Y_train = Resample_Data(X_train, Y_train, Sampling_Method)
    Sampled_Rows, Sampled_Pos = len(Y_train), sum(Y_train)
    del Data
    gc.collect()

    # -------------------------------------- Definiendo y ejecutando pipeline ------------------------------------------
    # Definiendo espacio de hiperparámetros del modelo
    Params = {'classifier__n_estimators': Integer(5, 10, 'uniform'),
              'classifier__learning_rate': Real(0.0, 0.2, 'uniform'),
              'classifier__max_depth': Integer(3, 4, 'uniform'),
              'classifier__colsample_bytree': Real(0.65, 0.80, 'uniform'),
              'classifier__reg_alpha': Real(0.1, 1.0, 'uniform')}
    # Instanciando y entrenando classifier
    XGB_Clf = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, tree_method='exact')
    Best_Model, Best_Params, Opt_Results = Train_Model(XGB_Clf, Params, X_train, Y_train, IsCloud)
    # Evaluando predicciones
    Y_pred_train = Best_Model.predict_proba(X_train)[:,1]
    Y_pred_test = Map_Probs(Best_Model.predict_proba(X_test)[:, 1], Init_Rows, Init_Pos, Sampled_Rows, Sampled_Pos)
    Tau = Find_Threshold(Y_pred_test, Init_Pos/Init_Rows)
    f1_train = f1_score(Y_train, Y_pred_train>=0.5)
    f1_test = f1_score(Y_test, Y_pred_test>=Tau)

    # ------------------------------------------- Escribiendo resultados -----------------------------------------------
    Best_Params['Init_Rows'], Best_Params['Init_Pos'] = Init_Rows, Init_Pos
    Best_Params['Sampled_Rows'], Best_Params['Sampled_Pos'], Best_Params['Tau']= Sampled_Rows, Sampled_Pos, Tau
    Write_Results(Model_Name, f1_train, f1_test, Best_Params, Results, pd.DataFrame(Opt_Results), OS)

# ----------------------------------------------------------------------------------------------------------------------
# Ejecución del cuerpo principal del "main"
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    end_time = time.time()
    print("Total Execution Time: {} seconds".format(end_time - start_time))
