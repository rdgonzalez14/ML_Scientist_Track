"""
------------------------------------------------------------------------------------------------------------------------
SCRIPT PARA LA PRUEBA INICIAL DE ACCESO MEDIANTE CODIGO CLIENTE A LOS SERVICIOS DE GCP
------------------------------------------------------------------------------------------------------------------------
Fecha: 13 de Mayo de 2020
Consideraciones:
    - Se asume que el entorno fue previamente configurado siguiendo la guía estipulada
    - El tiempo de espera es en SEGUNDOS
    - La cantidad de memoria RAM es el MB
Autor: Ruben D. Gonzalez R.
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importando librerías y/o funciones necesarias
#-----------------------------------------------------------------------------------------------------------------------
import googleapiclient.discovery
import pandas as pd
from GCP_Client.GCP_Functions import *

# ----------------------------------------------------------------------------------------------------------------------
# Inicializando variables
#-----------------------------------------------------------------------------------------------------------------------
# Definiendo variables generales
bool_switch = True
waiting_time = 60
# Definiendo propiedades de los servicios de GCP
project_id = 'tribal-environs-240721'
bucket_name = 'rd_ml_bucket'
image_name = 'base-image'
ncores = 8
nram = 131072
start_script = r"gs://rd_ml_bucket/Test/Scripts/GCP_Test_Script.py"

# ----------------------------------------------------------------------------------------------------------------------
# Cuerpo del "Main" del Script
#-----------------------------------------------------------------------------------------------------------------------
def main(project, bucket, start_script, image_name, bool_switch, waiting_time, ncores, nram, zone='us-central1-f',
         instance_name='demo-instance', wait=True):
    # Instanciando objeto de ComputeEngine
    compute = googleapiclient.discovery.build('compute', 'v1')
    # Creando instancia
    print('Creating instance.')
    operation = create_instance(compute, project, zone, ncores, nram, instance_name, bucket, start_script, image_name)
    wait_for_operation(compute, project, zone, operation['name'])
    instances = list_instances(compute, project, zone)
    print('Instances in project %s and zone %s:' % (project, zone))
    for instance in instances:
        print(' - ' + instance['name'])
    # Validando que el proceso del startup_script haya finalizado
    while bool_switch:
        try:
            DF = pd.read_csv(r"gs://rd_ml_bucket/Test/Results/DF_End_Logs.txt")
            bool_switch = False
        except:
            time.sleep(waiting_time)

    # Borrando instancia
    print('Deleting instance.')
    operation = delete_instance(compute, project, zone, instance_name)
    wait_for_operation(compute, project, zone, operation['name'])

if __name__ == '__main__':
    # Ejecutando funciones
    main(project_id, bucket_name, start_script, image_name, bool_switch, waiting_time, ncores, nram)
