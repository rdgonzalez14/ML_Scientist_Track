# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA OBTENCIÓN DEL MODELO BASE DEL CHALLENGE N°1
------------------------------------------------------------------------------------------------------------------------
Fecha: 14 de Junio de 2020
Consideraciones:
    - El código está diseñado para correr tanto en Windows como en Linux
    - Se standarizan features numéricos con escalas de magnitud diferentes
    - Se eliminan outliers extremos identificados en el EDA
    - Los features categóricos se codifican con OneHotEncoding
    - El modelo base es un xgboost tuneado
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
from sklearn.metrics import f1_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from Functions.General_Functions import *

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
start_time=time.time()
OS=platform.system()
IsCloud = 0
IsSearch = 0
Tune_Params = {'n_iter':[35, 250], 'n_points':[1, 5], 'n_jobs':[4, 20]}

# ----------------------------------------------------------------------------------------------------------------------
# Definición de funciones
#-----------------------------------------------------------------------------------------------------------------------
# --------------------------Función para definir pipelines para el entrenamiento del modelo-----------------------------
def Train_Model(clf, params, X_train,Y_train, X_test, Y_test, index, search):
    # Definiendo pipeline para features categóricos nominales
    categ_nom_transformer = Pipeline(steps=[('OneHot_encoding', OneHotEncoder(drop='first'))])
    # Definiendo el transformador de columnas para aplicar los pipelines de transformacion de features
    cat_nom_features = ['Card_Franchise', 'Status']
    preprocessor = ColumnTransformer(transformers=[('cat_nom', categ_nom_transformer, cat_nom_features)],
                                     remainder='passthrough')
    # Definiendo pipeline y ejecutando tuneo de hiperparámetros
    ML_Pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    Best_Score = 0
    for p_corte in np.arange(0.4, 0.65, 0.05):
        score = make_scorer(Score_Model, threshold=p_corte, greater_is_better=True, needs_threshold=True)
        Model = BayesSearchCV(ML_Pipe, params, scoring=score, cv=4, n_iter=Tune_Params['n_iter'][index],
                              n_jobs=Tune_Params['n_jobs'][index], n_points=Tune_Params['n_points'][index],
                              verbose=1).fit(X_train, Y_train)
        if search == 1:
            if Model.best_score_ > Best_Score:
                Best_Model, Best_Score, Best_Pcorte = Model, Model.best_score_, p_corte
        else:
            Y_pred_train, Y_pred_test = Model.predict_proba(X_train)[:, 1], Model.predict_proba(X_test)[:, 1]
            f1_train, f1_test = f1_score(Y_train, Y_pred_train >= p_corte), f1_score(Y_test, Y_pred_test >= p_corte)
            Write_Models('XGBoost_Base', f1_train, f1_test, Model.best_params_, p_corte, OS)
    if search == 1:
        return (Best_Model.best_estimator_, Best_Model.best_params_, Best_Model.cv_results_, Best_Pcorte)
    else:
        return (None, None, None, None)

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
    # Definiendo espacio de hiperparámetros del modelo
    Params = {'classifier__n_estimators': Integer(25, 50, 'uniform'),
              'classifier__learning_rate': Real(0.0, 0.25, 'uniform'),
              'classifier__max_depth': Integer(3, 4, 'uniform'),
              'classifier__colsample_bytree': Real(0.65, 0.85, 'uniform'),
              'classifier__reg_alpha': Real(0.1, 1.0, 'uniform')}
    # Instanciando y entrenando classifier
    XGB_Clf = xgb.XGBClassifier(objective='binary:logistic', n_jobs=-1, tree_method='exact')
    Best_Model, Best_Params, Opt_Results, P_corte = Train_Model(XGB_Clf, Params, X_train, Y_train, X_test, Y_test, IsCloud, IsSearch)
    if IsSearch == 1:
        # Evaluando predicciones
        Y_pred_train = Best_Model.predict_proba(X_train)[:,1]
        Y_pred_test = Best_Model.predict_proba(X_test)[:,1]
        f1_train = f1_score(Y_train, Y_pred_train>=P_corte)
        f1_test = f1_score(Y_test, Y_pred_test>=P_corte)
        # Escribiendo resultados
        Best_Params['P_corte'] = P_corte
        Write_Results('XGBoost_Base', f1_train, f1_test, Best_Params, Results, pd.DataFrame(Opt_Results), OS)

# ----------------------------------------------------------------------------------------------------------------------
# Ejecución del cuerpo principal del "main"
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
    end_time = time.time()
    print("Total Execution Time: {} seconds".format(end_time - start_time))
