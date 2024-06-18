import os
from utils.functions import *
from tqdm import tqdm
import boto3
from botocore.exceptions import NoCredentialsError
import argparse
from dotenv import load_dotenv
import cv2

# Crea un objeto ArgumentParser para definir argumentos y opciones para tu script
parser = argparse.ArgumentParser(description='Ejecutar script con un argumento de línea de comandos')

# Agrega un argumento posicional para el task_name
parser.add_argument('proceso', type=str, help='Nombre del proceso, opciones: new_task o get_data')
parser.add_argument('task_name', type=str, help='Nombre de la tarea, formato: levID-medID')

# Parsea los argumentos proporcionados por la línea de comandos
args = parser.parse_args()

# Usa el argumento task_name en tu programa
proceso = args.proceso
task_name = args.task_name

# Carga las variables del archivo .env
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_BUCKET = os.getenv('AWS_BUCKET')

# Parámetros de conexión
CVAT_HOST = os.getenv('CVAT_HOST')
CVAT_USERNAME = os.getenv('CVAT_USERNAME')
CVAT_PASSWORD = os.getenv('CVAT_PASSWORD')

# Conectar a CVAT
client = connect_to_cvat(CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD)

if proceso == 'get_img':
    
    # Buscar la tarea
    task = find_task(client, task_name)

    # Obtener el dataset con lables e imagenes
    get_dataset(task, CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD)

    # Correr unzip para descomprimir el archivo y que no haga print de lo que hace
    print(f"Descomprimiendo el archivo {task_name}.zip...")
    os.system(f'unzip {task_name}.zip -d {task_name} > /dev/null')
    print("Archivo descomprimido!")

    levID = task_name.split('-')[0]
    medID = task_name.split('-')[1]
    rootPath = f'{task_name}/obj_train_data/{levID}/{medID}/images'
    imagesPath = f'{task_name}/obj_train_data/{levID}/{medID}/images'
    lowImg = f'{task_name}/obj_train_data/{levID}/{medID}/img_mala_calidad'
    labelsPath = f'{task_name}/obj_train_data/{levID}/{medID}/labels'
    detectionsPath = f'{task_name}/obj_train_data/{levID}/{medID}/detections'
    s3_labels = f"{levID}/{medID}/labels"
    s3_detections = f"{levID}/{medID}/detections"
    s3_lowImg = f"{levID}/{medID}/img_mala_calidad"
    s3_reporte = f'{levID}/{medID}'
    filenames = os.listdir(rootPath)   
    
    os.makedirs(lowImg, exist_ok=True)
    images = os.listdir(imagesPath)

    for filename in tqdm(images, desc="Bajando calidad IMG"):
        if filename.endswith('.JPG'):
 
            image_path = os.path.join(imagesPath, filename)
            imgData = cv2.imread(image_path)
            imgResized = cv2.resize(imgData, (870, 650))

            cv2.imwrite(os.path.join(lowImg,filename), imgResized)
        


    s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
    )     
        
      # Recorrer todos los archivos en la carpeta local
    for root, dirs, files in os.walk(lowImg):
        for file in tqdm(files, desc="Subiendo archivos a S3"):
                # Ruta completa al archivo
                local_path = os.path.join(root, file)

                # Generar la clave para el S3
                s3_key = os.path.relpath(local_path, lowImg)  # Obtener la ruta relativa desde la carpeta local
                s3_full_key = os.path.join(s3_lowImg, s3_key)

                # Subir el archivo al bucket S3
                try:
                    s3.upload_file(local_path, AWS_BUCKET, s3_full_key)
                    
                except NoCredentialsError:
                    print("No se encontraron las credenciales para AWS.")
                except Exception as e:
                    print(f"Error al subir el archivo {file}: {str(e)}")   
        
        
    
        
    
    # Eliminar el archivo zip y la carpeta descomprimida
    os.system(f'rm -r {task_name}')
    os.system(f'rm {task_name}.zip')
    print("Archivos locales eliminados!")
    print("Proceso finalizado!")          
    
