# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
MÓDULO CON FUNCIONES Y CLASES DE PROPÓSITO GENERAL
------------------------------------------------------------------------------------------------------------------------
Fecha: 14 de Junio de 2020
Consideraciones:
    - El comentario de encabezado describe la funcionalidad de cada método/clase
    - BMR es el Bayes Minimum Risk Theorem, utilizado para mapear (calibrar) entre U.Balanceado y U.Desbalanceado
    - El Tau de corte se calcula a partir de la probabilidad de ser positivo en el U.Desbalanceado
Autor: Ruben D. Gonzalez R.
"""
# ----------------------------------------------------------------------------------------------------------------------
# Importando librerías y/o funciones necesarias
#-----------------------------------------------------------------------------------------------------------------------
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ----------------------------------------------------------------------------------------------------------------------
# Definición de las funciones y clases
#-----------------------------------------------------------------------------------------------------------------------
# ------------------------------Función para levantar datasets desde los archivos ".csv"--------------------------------
def Load_Dataset(OS):
    # Levantando datasets
    if OS == 'Windows':
        csvs = glob.glob(r".\Datasets\*.txt")
        # Levantando DataFrame de mejores resultados
        try:
            Results = pd.read_csv(r".\Results\Best_Results.csv", sep=',', index_col=None, header=0)
        except:
            Results = None
    else:
        path = r"gs://rd_ml_bucket/ML_CN1/Datasets/"
        csvs = [path+"Train_p1.txt", path+"Train_p2.txt"]
        # Levantando DataFrame de mejores resultados
        try:
            Results = pd.read_csv(r"gs://rd_ml_bucket/ML_CN1/Results/Best_Results.csv", sep=',', index_col=None, header=0)
        except:
            Results = None
    Dataset = pd.DataFrame()
    for csv in csvs:
        Dataset = Dataset.append(pd.read_csv(csv, sep=',', index_col=None, header=0))
    return (Dataset.reset_index(drop=True), Results)

# ------------------------------Función para eliminar outliers extremos identificados-----------------------------------
def Drop_Outliers(Dataset):
    # Identificando y eliminando los outliers de las columnas numéricas
    Num_Dataset = Dataset.loc[:, ~Dataset.columns.isin(['Card_Franchise', 'Status'])]
    Outliers_Rows = ((Num_Dataset>50)|(Num_Dataset<-60)).any(axis=1)
    Dataset = Dataset.loc[~Outliers_Rows,:]
    return Dataset

# ------------------------------- Función para realizar scoring de un modelo fitteado ----------------------------------
def Score_Model(y_true, y_pred, threshold):
    return f1_score(y_true, y_pred >= threshold)

# -----------------------------------Función para registrar resultados obtenidos ---------------------------------------
def Write_Results(name, f1_train, f1_test, Params, Results, Search_Params, OS):
    # Generando DF one-liner de salida
    New_Results = pd.DataFrame({'Modelo': name, 'Train_Score': f1_train, 'Test_Score': f1_test,
                                'Params': str(Params)}, index=[0])
    if OS == 'Windows':
        path = r".\Results\Best_Results.csv"
        path_params = r".\Results\Searched_Params.csv"
    else:
        path = r"gs://rd_ml_bucket/ML_CN1/Results/Best_Results.csv"
        path_params = r"gs://rd_ml_bucket/ML_CN1/Results/Searched_Params.csv"
    if Results is not None:
        Results = Results.append(New_Results)
        Results.to_csv(path, sep=",", header=True, index=False)
    else:
        New_Results.to_csv(path, sep=",", header=True, index=False)
    try:
        Search_Params.to_csv(path_params, sep=",", header=True, index=False)
    except:
        pass

# ---------------------------------Función para registrar mejor modelo por p_corte -------------------------------------
def Write_Models(name, f1_train, f1_test, Params, p_corte, OS):
    # Generando DF one-liner de salida
    New_Results = pd.DataFrame({'Modelo': name, 'Train_Score': f1_train, 'Test_Score': f1_test,
                                'Params': str(Params), 'P_corte' :p_corte}, index=[0])
    if OS == 'Windows':
        path = r".\Results\Best_Models.csv"
    else:
        path = r"gs://rd_ml_bucket/ML_CN1/Results/Best_Models.csv"
    try:
        DF = pd.read_csv(path, sep=',', index_col=None, header=0)
        DF = DF.append(New_Results)
        DF.to_csv(path, sep=",", header=True, index=False)
    except:
        New_Results.to_csv(path, sep=",", header=True, index=False)

# ------------------------------Función para calibrar y mapear probabilidades usando BMR--------------------------------
def Map_Probs(data, train_pop, target_pop, sampled_train_pop, sampled_target_pop):
    calibrated_data = \
    ((data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)) /
    ((
        (1 - data) * (1 - target_pop / train_pop) / (1 - sampled_target_pop / sampled_train_pop)
     ) +
     (
        data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)
     )))
    return calibrated_data

# --------------------------------Función para encontrar p_corte en el U.Desbalanceado----------------------------------
def Find_Threshold(probs, pos_prob):
    probs, lower_limit = np.sort(probs), 1-pos_prob
    threshold = probs[int(len(probs)*lower_limit)]
    return threshold
