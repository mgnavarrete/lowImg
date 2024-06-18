import os
import zipfile
from utils.functions import *
from tqdm import tqdm
import boto3
from botocore.exceptions import NoCredentialsError
import argparse
from dotenv import load_dotenv
import cv2
import shutil

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

    # Correr unzip para descomprimir el archivo
    print(f"Descomprimiendo el archivo {task_name}.zip...")
    with zipfile.ZipFile(f'{task_name}.zip', 'r') as zip_ref:
        zip_ref.extractall(task_name)
    
    print("Archivo descomprimido!")

    rootPath = os.path.join(task_name, 'obj_train_data', task_name.split('-')[0], task_name.split('-')[1], 'images')
    lowImg = os.path.join(task_name, 'obj_train_data', task_name.split('-')[0], task_name.split('-')[1], 'img_mala_calidad')

    os.makedirs(lowImg, exist_ok=True)
    images = os.listdir(rootPath)

    for filename in tqdm(images, desc="Bajando calidad IMG"):
        if filename.endswith('.JPG'):
            image_path = os.path.join(rootPath, filename)
            imgData = cv2.imread(image_path)
            imgResized = cv2.resize(imgData, (870, 650))

            cv2.imwrite(os.path.join(lowImg, filename), imgResized)

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

    # Recorrer todos los archivos en la carpeta local
    for root, dirs, files in os.walk(lowImg):
        for file in tqdm(files, desc="Subiendo archivos a S3"):
            local_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_path, lowImg)
            s3_full_key = os.path.join(task_name.split('-')[0], task_name.split('-')[1], 'img_mala_calidad', s3_key)

            try:
                s3.upload_file(local_path, AWS_BUCKET, s3_full_key)
            except NoCredentialsError:
                print("No se encontraron las credenciales para AWS.")
            except Exception as e:
                print(f"Error al subir el archivo {file}: {str(e)}")   

    # Eliminar el archivo zip y la carpeta descomprimida
    shutil.rmtree(task_name)
    os.remove(f'{task_name}.zip')
    print("Archivos locales eliminados!")
    print("Proceso finalizado!")
