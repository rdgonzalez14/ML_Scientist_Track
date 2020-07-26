# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA OBTENCIÓN DEL ENSEMBLE DE LOS MEJORES MODELOS DEL CHALLENGE N°1
------------------------------------------------------------------------------------------------------------------------
Fecha: 07 de Julio de 2020
Consideraciones:
    - El código está diseñado para correr tanto en Windows como en Linux
    - Se obtienen las configuraciones de los mejores modelos desde el registro de mejores resultados
    - Se genera un VotingClassifier a partir de los mejores modelos obtenidos desde el registro
    - Se realiza la predicción del ensemble mediante Hard-Voting mayoritario
Autor: Ruben D. Gonzalez R.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importando librerías y/o funciones necesarias
#-----------------------------------------------------------------------------------------------------------------------
import time, ast, platform, gc, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import scale, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from Functions.General_Functions import *

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
start_time=time.time()
OS=platform.system()
IsCloud = 0

# ----------------------------------------------------------------------------------------------------------------------
# Definición de funciones y clases
#-----------------------------------------------------------------------------------------------------------------------
# ------------------------Función para levantar archivo con parámetros de los mejores modelos---------------------------
def Load_Best_Models(OS):
    if OS == 'Windows':
        Models = pd.read_csv(r".\Results\Best_Models.csv", sep=',', index_col=None, header=0)
    else:
        Models = pd.read_csv(r"gs://rd_ml_bucket/ML_CN1/Results/Best_Models.csv", sep=',', index_col=None, header=0)
    return Models

# --------------------------Clase para instanciar Voting Classifier a partir de los modelos-----------------------------
class VotingClassifier:

    def __init__(self, modelos):
        # Iterando sobre los mejores modelos
        classifiers = []
        for model in modelos:
            params = dict(ast.literal_eval(model))
            XGB_Clf = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, tree_method='exact', **params)
            classifiers.append(XGB_Clf)
        self.clfs = classifiers

    def fit(self, X_train,Y_train):
        # Definiendo pipeline para features categóricos nominales
        categ_nom_transformer = Pipeline(steps=[('OneHot_encoding', OneHotEncoder(drop='first'))])
        # Definiendo el transformador de columnas para aplicar los pipelines de transformacion de features
        cat_nom_features = ['Card_Franchise', 'Status']
        preprocessor = ColumnTransformer(transformers=[('cat_nom', categ_nom_transformer, cat_nom_features)],
                                         remainder='passthrough')
        # Definiendo pipeline y entrenando modelos
        self.models = []
        for clf in self.clfs:
            ML_Pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)]).fit(X_train, Y_train)
            self.models.append(ML_Pipe)
        return self

    def predict(self, X, p_cortes):
        votes = pd.DataFrame()
        # Obteniendo la predicción de cada modelo según su p_corte
        for modelo, p_corte in zip(self.models, p_cortes):
            y_pred = modelo.predict_proba(X)[:,1] >= p_corte
            votes[p_corte] = y_pred
        # Obteniendo prediccion del ensemble
        sum_votes = votes.sum(axis=1)>=3
        return sum_votes

# ----------------------------------------------------------------------------------------------------------------------
# Función con el cuerpo principal del "main"
#-----------------------------------------------------------------------------------------------------------------------
def main():
    # ---------------------------- Levantando datasets y modelos desde los archivos ".csv" -----------------------------
    Data, Results = Load_Dataset(OS)
    Modelos = Load_Best_Models(OS)

    # --------------------------------------- Manipulando y preparando datasets ----------------------------------------
    Modelos['Params'] = Modelos['Params'].str.slice(start=11).replace(to_replace=['\(\[', '\)\]','classifier__'], value='', regex=True)
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

    # ---------------------------------- Instanciando y entrenando Voting Classifier -----------------------------------
    VotClf= VotingClassifier(Modelos['Params']).fit(X_train, Y_train)
    # Evaluando predicciones
    y_pred_train = VotClf.predict(X_train, Modelos['P_corte'])
    y_pred_test = VotClf.predict(X_test,Modelos['P_corte'])
    f1_train = f1_score(Y_train, y_pred_train)
    f1_test = f1_score(Y_test, y_pred_test)

    # ------------------------------------------- Escribiendo resultados -----------------------------------------------
    Write_Results('XGBoost_Ensemble', f1_train, f1_test, None, Results, None, OS)

    # --------------------------------------Entrenando y guardando modelo final ----------------------------------------
    DefVotClf = VotingClassifier(Modelos['Params']).fit(Data, Label)
    y_pred_train_def = DefVotClf.predict(Data, Modelos['P_corte'])
    f1_train_def = f1_score(Label, y_pred_train_def)
    # Guardando modelo en disco
    filename = r".\Results\FinalModel_CN1.sav"
    pickle.dump(DefVotClf, open(filename, 'wb'))

# ----------------------------------------------------------------------------------------------------------------------
# Ejecución del cuerpo principal del "main"
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    end_time = time.time()
    print("Total Execution Time: {} seconds".format(end_time - start_time))
