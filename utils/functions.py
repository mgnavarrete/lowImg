from cvat_sdk import Client
from cvat_sdk.api_client import Configuration, ApiClient
from time import sleep
import json


def read_metadata(file_path):
    with open(file_path, 'r') as file:
        metadata = json.load(file)
    return metadata

def connect_to_cvat(host, username, password):
    """Conecta al servidor CVAT y retorna el objeto de cliente."""
    client = Client(host)
    client.login((username, password))
    return client


def find_task(client, task_name):
    """Busca una tarea por nombre y retorna la primera coincidencia."""
    tasks = client.tasks.list()
    for task in tasks:
        if task.name == task_name:
            print(f"Tarea encontrada: {task_name}")
            return task
        
    if not tasks:
        raise ValueError(f"No se encontró la tarea con el nombre '{task_name}'")
    return tasks[0]

def get_dataset(task, CVAT_HOST, CVAT_USERNAME, CVAT_PASSWORD):
    """Obtiene el nombre del archivo de imagen para un marco específico."""
    configuration = Configuration(
    host = CVAT_HOST,
    username = CVAT_USERNAME,
    password = CVAT_PASSWORD,
)
    with ApiClient(configuration) as api_client:
        # Export a task as a dataset
        while True:
            (_, response) = api_client.tasks_api.retrieve_dataset(
                id=task.id,
                format='YOLO 1.1',
                _parse_response=False,
            )
            if response.status == 201:
                print("Respuesta del CVAT recibida!")
                break
            
            print("Esperando la respuesta del servidor CVAT...")
            sleep(5)
            

        (_, response) = api_client.tasks_api.retrieve_dataset(
            id=task.id,
            format='YOLO 1.1',
            action="download",
            _parse_response=False,
         )
        
        # Save the resulting file
        with open(f'{task.name}.zip', 'wb') as output_file:
            print(f"Comenzando a descargar {task.name}.zip...")
            output_file.write(response.data)
            print(f"{task.name}.zip descargado!")
               
                