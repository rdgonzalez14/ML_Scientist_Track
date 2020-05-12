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

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
#  Levantando el dataset desde el archivo csv del bucket e imprimiendolo en la consola
DF = pd.read_csv(r"gs://rd_ml_bucket/Test/MLC_Members.txt")
print(DF)
print("\n La potencia y el poder del Cloud Computing están en mis manos!!",
      "\n ... y un gran poder conlleva a una gran responsabilidad.")