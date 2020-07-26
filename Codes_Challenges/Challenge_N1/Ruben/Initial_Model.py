# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA OBTENCIÓN DEL MODELO BASE INICIAL DEL CHALLENGE N°1
------------------------------------------------------------------------------------------------------------------------
Fecha: 14 de Junio de 2020
Consideraciones:
    - El código está diseñado para correr tanto en Windows como en Linux
    - Se standarizan features numéricos con escalas de magnitud diferentes
    - Se eliminan outliers extremos identificados en el EDA
    - Los features categóricos se codifican con OneHotEncoding
    - El modelo base es un xgboost sin tunear
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
from sklearn.preprocessing import scale, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from Functions.General_Functions import *

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
start_time=time.time()
OS=platform.system()
IsExecution = 1

# ----------------------------------------------------------------------------------------------------------------------
# Definición de funciones
#-----------------------------------------------------------------------------------------------------------------------
# --------------------------Función para definir pipelines para el entrenamiento del modelo-----------------------------
def Train_Model(clf, X_train,Y_train):
    # Definiendo pipeline para features categóricos nominales
    categ_nom_transformer = Pipeline(steps=[('OneHot_encoding', OneHotEncoder(drop='first'))])
    # Definiendo el transformador de columnas para aplicar los pipelines de transformacion de features
    cat_nom_features = ['Card_Franchise', 'Status']
    preprocessor = ColumnTransformer(transformers=[('cat_nom', categ_nom_transformer, cat_nom_features)],
                                     remainder='passthrough')
    # Definiendo y ejecutando pipeline de entrenamiento del modelo
    ML_Pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    ML_Pipe.fit(X_train,Y_train)
    return ML_Pipe

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
    del Data
    gc.collect()

    # -------------------------------------- Definiendo y ejecutando pipeline ------------------------------------------
    # Instanciando y entrenando classifier
    if IsExecution == 0:
        Params = {'objective':'binary:logistic', 'max_depth':5, 'learning_rate':0.2, 'n_jobs':-1, 'n_estimators':200, 'verbose':1}
    else:
        Params = {'objective': 'binary:logistic', 'n_jobs': -1, 'max_depth': 4, 'learning_rate': 0.25, 'n_estimators': 47,
                  'colsample_bytree': 0.80, 'reg_alpha': 0.16, 'tree_method': 'exact','verbose': 1}
    XGB_Clf = xgb.XGBClassifier(**Params)
    Model = Train_Model(XGB_Clf, X_train, Y_train)
    # Evaluando predicciones
    Y_pred_train = Model.predict_proba(X_train)[:,1]
    Y_pred_test = Model.predict_proba(X_test)[:,1]
    f1_train = f1_score(Y_train, Y_pred_train>0.5)
    f1_test = f1_score(Y_test, Y_pred_test>0.5)
    # Analizando probabilidades de los positivos
    Train_Array = np.column_stack((Y_train, Y_pred_train))
    Train_Pos_Array = Train_Array[Train_Array[:,0]==1]
    Test_Array = np.column_stack((Y_test, Y_pred_test))
    Test_Pos_Array = Test_Array[Test_Array[:,0] == 1]

    # ------------------------------------------- Escribiendo resultados -----------------------------------------------
    Write_Results('XGBoost_Initial', f1_train, f1_test, Params, Results)

# ----------------------------------------------------------------------------------------------------------------------
# Ejecución del cuerpo principal del "main"
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    end_time = time.time()
    print("Total Execution Time: {} seconds".format(end_time - start_time))
