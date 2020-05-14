#! /home/rubend1493/anaconda3/bin/python3
"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA LA PRUEBA INICIAL DE CLOUD COMPUTING EN EL ENTORNO DE GCP
------------------------------------------------------------------------------------------------------------------------
Fecha: 12 de Mayo de 2020
Consideraciones:
    - Se asume que el entorno fue previamente configurado siguiendo la guía estipulada
Autor: Ruben D. Gonzalez R.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importando librerías y/o funciones necesarias
#-----------------------------------------------------------------------------------------------------------------------
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
#  Levantando el dataset desde el archivo csv del bucket e imprimiendolo en la consola
DF = pd.read_csv(r"gs://rd_ml_bucket/Test/Datasets/MLC_Members.txt")
print(DF)
print("\n La potencia y el poder del Cloud Computing están en mis manos!!",
      "\n ... y un gran poder conlleva a una gran responsabilidad.")

# ----------------------------------------------------------------------------------------------------------------------
# Ejecutando proceso y escribiendo resultados en el bucket
#-----------------------------------------------------------------------------------------------------------------------
# Escribiendo resultados en el bucket
DF.to_csv(r"gs://rd_ml_bucket/Test/Results/DF_MLC_Members.txt", index = False)
# Escribiendo DF con loggs de finalización
df_end_logs = pd.DataFrame({'Id':1,'Name':'Process','Status':'Completed','End_Time':datetime.now().strftime("%d/%m/%Y %H:%M:%S")},
                           index=[0])
df_end_logs.to_csv(r"gs://rd_ml_bucket/Test/Results/DF_End_Logs.txt", index = False)