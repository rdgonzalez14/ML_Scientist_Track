# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA EJECUTAR EL MEJOR MODELO OBTENIDO DEL CHALLENGE N°1
------------------------------------------------------------------------------------------------------------------------
Fecha: 07 de Julio de 2020
Consideraciones:
    - El código está diseñado para correr tanto en Windows como en Linux
    - Se levanta el objeto serializado en pickle del mejor modelo y se utiliza para hacer predicciones
Autor: Ruben D. Gonzalez R.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importando librerías y/o funciones necesarias
#-----------------------------------------------------------------------------------------------------------------------
import time, ast, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import scale, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from Functions.General_Functions import *
from Ensemble_Models import VotingClassifier

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
start_time=time.time()
Model_File = r".\Results\FinalModel_CN1.sav"

# ----------------------------------------------------------------------------------------------------------------------
# Función con el cuerpo principal del "main"
#-----------------------------------------------------------------------------------------------------------------------
def main():
    # ---------------------------- Levantando datasets y modelos desde los archivos ".csv" -----------------------------
    csv = glob.glob(r".\Datasets\Test*.csv")
    Data = pd.read_csv(csv[0], sep=',', index_col=None, header=0)
    Modelos = pd.read_csv(r".\Results\Best_Models.csv", sep=',', index_col=None, header=0)
    DefVotClf = pickle.load(open(Model_File, 'rb'))

    # --------------------------------------- Manipulando y preparando dataset -----------------------------------------
    Modelos['Params'] = Modelos['Params'].str.slice(start=11).replace(to_replace=['\(\[', '\)\]','classifier__'], value='', regex=True)
    # Estandarizando variables con magnitudes elevadas y eliminando las originales
    Data["Time_scaled"] = scale(Data["Time"], with_mean=True, with_std=True)
    Data["Amount_scaled"] = scale(Data["Amount"], with_mean=True, with_std=True)
    Data["Card_Limit_scaled"] = scale(Data["Card_Limit"], with_mean=True, with_std=True)
    Data.drop(columns=['Time','Amount','Card_Limit'], inplace=True)
    # Separando features del target
    Label = Data['Class']
    Data.drop(columns='Class', inplace=True)

    # ---------------------------------- Ejecutando modelo para obtener predicciones -----------------------------------
    y_pred_eval = DefVotClf.predict(Data, Modelos['P_corte'])
    f1_eval = f1_score(Label, y_pred_eval)
    print("El F1 score en el dataset de evaluación es de: {0}".format(f1_eval))

# ----------------------------------------------------------------------------------------------------------------------
# Ejecución del cuerpo principal del "main"
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    end_time = time.time()
    print("Total Execution Time: {} seconds".format(end_time - start_time))
