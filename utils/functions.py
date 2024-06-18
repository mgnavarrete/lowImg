import os
from cvat_sdk import Client
from cvat_sdk.api_client import Configuration, ApiClient
from time import sleep
from PIL import Image, ImageDraw
from collections import defaultdict
from random import randint
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

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


# Función para dibujar bounding boxes y sobrescribir la imagen original
def draw_bounding_boxes(image_path, label_path):
    # Cargar la imagen y obtener sus dimensiones originales
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size  # Dimensiones originales de la imagen

    # Leer el archivo de etiquetas
    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            labels = file.readlines()

        for label in labels:
            # Formato YOLO: <class_id> <x_center> <y_center> <width> <height>
            label_info = label.split()

            x_center = float(label_info[1]) * image_width  # Centro X desnormalizado
            y_center = float(label_info[2]) * image_height  # Centro Y desnormalizado
            box_width = float(label_info[3]) * image_width  # Ancho desnormalizado
            box_height = float(label_info[4]) * image_height  # Altura desnormalizada

            # Calcular las coordenadas del bounding box
            x_min = int(x_center - box_width / 2)
            x_max = int(x_center + box_width / 2)
            y_min = int(y_center - box_height / 2)
            y_max = int(y_center + box_height / 2)

            # Dibujar el bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)

    # Sobrescribir la imagen original con el bounding box
    image.save(image_path)

# Funcion para crear la base del reporte
def create_base_report():
    report_dict = defaultdict(dict)
    nClasses = 3
    for id in range(nClasses):
        if id == 0:
            type = "antenna"
            model = "A01-2"
        elif id == 1:
            type = "antenna"
            model = "RRU"
        elif id == 2:
            type = "Micro Wave"
            model = "MW"    
                    
        report_dict[id] = {
            "type": type,
            "model": model,
            "detected": 0,
            "filenames": list(),
        }

    return report_dict


# Variables globales para almacenar los puntos de clic
clicked_points = []

def on_click(event):
    global clicked_points
    if event.inaxes:
        clicked_point = (event.xdata, event.ydata)
        clicked_points.append(clicked_point)
        plt.gca().add_patch(plt.Circle(clicked_point, radius=10, color='red'))
        plt.draw()

def select_tube_points(imageCenital, tube_distance_cm):
    global clicked_points
    clicked_points = []

    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(10, 10))
    mng = plt.get_current_fig_manager()


    # Mostrar la imagen cenital
    plt.imshow(cv2.cvtColor(imageCenital, cv2.COLOR_BGR2RGB))
    plt.title('Seleccionar dos puntos de ref')
    plt.axis('off')

    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan dos clics
    while len(clicked_points) < 2:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    # Calcular la distancia en píxeles del tubo
    tube_distance_px = np.linalg.norm(np.array(clicked_points[0]) - np.array(clicked_points[1]))


    # Relación píxeles a cm
    px_to_cm = tube_distance_cm / tube_distance_px

    print(f"Relación píxeles a cm: {px_to_cm:.4f} cm/px")
    return px_to_cm

def getAntenaCM(imageFrontal, pix2cm):
    global clicked_points
    clicked_points = []

    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(10, 10))
    mng = plt.get_current_fig_manager()


    # Mostrar la imagen cenital
    plt.imshow(cv2.cvtColor(imageFrontal, cv2.COLOR_BGR2RGB))
    plt.title('Seleccionar dos puntos de ref')
    plt.axis('off')

    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan dos clics
    while len(clicked_points) < 2:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()


    distPix = np.linalg.norm(np.array(clicked_points[0]) - np.array(clicked_points[1]))
    
    # Calcular la distancia en cm 
    distCm = distPix * pix2cm
    
    return distCm

def calculate_angle_and_width(imageCenital, imageFrontal, yawDegreesCenital, px_to_cm):
    global clicked_points
    clicked_points = []

    # Dibujar en Imagen Cenital en norte SEGUN EL YAW
    cv2.line(imageCenital, (imageCenital.shape[1] // 2, imageCenital.shape[0] // 2), (imageCenital.shape[1] // 2, 0), (0, 0, 255), 2)

    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(20, 10))
    mng = plt.get_current_fig_manager()


     # Mostrar la imagen frontal
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imageFrontal, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Frontal')
    plt.axis('off')

    # Mostrar la imagen cenital
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imageCenital, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Cenital')
    plt.axis('off')
   

    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan dos clics (para la antena)
    while len(clicked_points) < 2:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    # Calcular el punto medio de los puntos de la antena
    antenna_midpoint = ((clicked_points[0][0] + clicked_points[1][0]) / 2,
                        (clicked_points[0][1] + clicked_points[1][1]) / 2)

    # Calcular el ancho de la antena en píxeles y convertir a cm
    antenna_width_px = np.linalg.norm(np.array(clicked_points[0]) - np.array(clicked_points[1]))
    antenna_width_cm = antenna_width_px * px_to_cm

    # Centro de la imagen cenital
    h, w, _ = imageCenital.shape
    cx, cy = w / 2, h / 2

    # Coordenadas del punto medio de la antena
    px, py = antenna_midpoint

    # Diferencia en las coordenadas
    dx = px - cx
    dy = cy - py  # Nota: invertimos la coordenada y porque la imagen tiene el origen en la esquina superior izquierda

    # Calcular el ángulo entre el centro y el punto clicado
    alpha = np.arctan2(dy, dx)

    # Ajustar el ángulo según el yaw
    yaw_rad = np.radians(yawDegreesCenital)
    alpha_prime = alpha - yaw_rad

    # Convertir el ángulo ajustado a grados
    alpha_prime_deg = np.degrees(alpha_prime)

    # Normalizar el ángulo para que esté en el rango [0, 360)
    angle = (alpha_prime_deg + 360) % 360

    return round(angle,1), round(antenna_width_cm,1)


def calculate_angle(imageCenital, imageFrontal, yawDegreesCenital):
    global clicked_points
    clicked_points = []

    # Dibujar en Imagen Cenital en norte SEGUN EL YAW
    cv2.line(imageCenital, (imageCenital.shape[1] // 2, imageCenital.shape[0] // 2), (imageCenital.shape[1] // 2, 0), (0, 0, 255), 2)

    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(20, 10))
    mng = plt.get_current_fig_manager()


     # Mostrar la imagen frontal
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imageFrontal, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Frontal')
    plt.axis('off')

    # Mostrar la imagen cenital
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imageCenital, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Cenital')
    plt.axis('off')
   

    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan dos clics (para la antena)
    while len(clicked_points) < 1:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    # Calcular el punto medio de los puntos de la antena
    antenna_midpoint = clicked_points[0]

    # Centro de la imagen cenital
    h, w, _ = imageCenital.shape
    cx, cy = w / 2, h / 2

    # Coordenadas del punto medio de la antena
    px, py = antenna_midpoint

    # Diferencia en las coordenadas
    dx = px - cx
    dy = cy - py  # Nota: invertimos la coordenada y porque la imagen tiene el origen en la esquina superior izquierda

    # Calcular el ángulo entre el centro y el punto clicado
    alpha = np.arctan2(dy, dx)

    # Ajustar el ángulo según el yaw
    yaw_rad = np.radians(yawDegreesCenital)
    alpha_prime = alpha - yaw_rad

    # Convertir el ángulo ajustado a grados
    alpha_prime_deg = np.degrees(alpha_prime)

    # Normalizar el ángulo para que esté en el rango [0, 360)
    angle = (alpha_prime_deg + 360) % 360


    return round(angle,1)

def calculate_angleOnly(imageCenital, yawDegreesCenital):
    global clicked_points
    clicked_points = []

    # Dibujar en Imagen Cenital en norte SEGUN EL YAW
    cv2.line(imageCenital, (imageCenital.shape[1] // 2, imageCenital.shape[0] // 2), (imageCenital.shape[1] // 2, 0), (0, 0, 255), 2)

    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(20, 10))
    mng = plt.get_current_fig_manager()


    plt.imshow(cv2.cvtColor(imageCenital, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Cenital')
    plt.axis('off')
   

    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan dos clics (para la antena)
    while len(clicked_points) < 1:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    # Calcular el punto medio de los puntos de la antena
    antenna_midpoint = clicked_points[0]

    # Centro de la imagen cenital
    h, w, _ = imageCenital.shape
    cx, cy = w / 2, h / 2

    # Coordenadas del punto medio de la antena
    px, py = antenna_midpoint

    # Diferencia en las coordenadas
    dx = px - cx
    dy = cy - py  # Nota: invertimos la coordenada y porque la imagen tiene el origen en la esquina superior izquierda

    # Calcular el ángulo entre el centro y el punto clicado
    alpha = np.arctan2(dy, dx)

    # Ajustar el ángulo según el yaw
    yaw_rad = np.radians(yawDegreesCenital)
    alpha_prime = alpha - yaw_rad

    # Convertir el ángulo ajustado a grados
    alpha_prime_deg = np.degrees(alpha_prime)

    # Normalizar el ángulo para que esté en el rango [0, 360)
    angle = (alpha_prime_deg + 360) % 360


    return round(angle,1)



def hightTower(imageFrontal):
    global clicked_points
    clicked_points = []
    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(20, 10))
    mng = plt.get_current_fig_manager()



    plt.imshow(cv2.cvtColor(imageFrontal, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Frontal')
    plt.axis('off')

    
    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan tres clics (para la antena)
    while len(clicked_points) < 1:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    altoTorre = (clicked_points[0])
    return altoTorre
    
def pointAntena(imageFrontal, cmAntena):
    global clicked_points
    clicked_points = []

    # Crear una figura en pantalla completa
    fig = plt.figure(figsize=(10, 10))
    mng = plt.get_current_fig_manager()


    # Mostrar la imagen cenital
    plt.imshow(cv2.cvtColor(imageFrontal, cv2.COLOR_BGR2RGB))
    plt.title('Seleccionar altura antena')
    plt.axis('off')

    # Conectar el evento de clic
    cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)

    # Esperar hasta que se hagan dos clics
    while len(clicked_points) < 2:
        plt.pause(0.1)

    # Desconectar el evento de clic
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    # Calcular la distancia en píxeles del tubo
    print(clicked_points)
    tube_distance_px = np.linalg.norm(np.array(clicked_points[0]) - np.array(clicked_points[1]))
    
    print(tube_distance_px)
    # Medida real del tubo en cm (reemplaza esto con el valor conocido)
    tube_distance_cm = cmAntena

    # Relación píxeles a cm
    px_to_cm = tube_distance_cm / tube_distance_px

    print(f"Relación píxeles a cm: {px_to_cm:.4f} cm/px")
    
    punto_medio = ((clicked_points[0][0] + clicked_points[1][0]) / 2,
                        (clicked_points[0][1] + clicked_points[1][1]) / 2)
    return px_to_cm, punto_medio


def calculate_azimuth(x_center, y_center, yaw_degree, image_width, image_height):
    # Coordenadas del centro de la imagen
    image_center = np.array([image_width / 2, image_height / 2])
    
    # Coordenadas del punto de interés
    point_center = np.array([x_center, y_center])
    
    # Desplazamiento desde el centro
    delta = point_center - image_center
    
    # Calcular el ángulo en radianes y luego convertir a grados
    angle_radians = np.arctan2(delta[1], delta[0])
    angle_degrees = np.degrees(angle_radians)
    
    # Ajustar con el ángulo de yaw del dron y asegurarse de que el norte es 0
    azimuth = (angle_degrees + yaw_degree) % 360
    
    # Convertir azimut a un rango de 0 a 360 grados
    if azimuth < 0:
        azimuth += 360
    
    return azimuth

def drawbbox(imageFrontal, label_info, yaw_degree):# dibujar el bounding box in imageFrontal
    x_center = float(label_info[1]) * imageFrontal.shape[1]
    y_center = float(label_info[2]) * imageFrontal.shape[0]
    box_width = float(label_info[3]) * imageFrontal.shape[1]
    box_height = float(label_info[4]) * imageFrontal.shape[0]
    x_min = int(x_center - box_width / 2)
    x_max = int(x_center + box_width / 2)
    y_min = int(y_center - box_height / 2)
    y_max = int(y_center + box_height / 2)
    cv2.rectangle(imageFrontal, (x_min, y_min), (x_max, y_max), (0, 0, 255), 10)

    
    return imageFrontal
# Funcion que saca infromacion de label y rellena el reporte 
def get_report(label_path, report_dict, IDAntena, imageCenital, imageFrontal, yawDegreesCenital, yawDegreesFrontal, px_to_cm, getWith = False):    
    import cv2
    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            labels = file.readlines()
        if len(labels) == 0:
            return IDAntena, report_dict
        else:
            for label in labels:
                IDAntena += 1
                # Formato YOLO: <class_id> <x_center> <y_center> <width> <height>
                label_info = label.split()
                imageFrontalData = cv2.imread(imageFrontal)
                imageWidth = imageFrontalData.shape[1]
                imageHeight = imageFrontalData.shape[0]
                imageBBOX = drawbbox(imageFrontalData, label_info, yawDegreesFrontal)
                modeloTipo = { 0:"RF", 1:"RRU", 2:"Micro Wave" }
                width = None
                height = None
                angle = None
                # Ancho y alto de bbox
                
                
                if getWith:
                    if True:
                        angle, width = calculate_angle_and_width(imageCenital, imageBBOX, yawDegreesCenital, px_to_cm)
                        widthbbox = float(label_info[3]) * imageWidth
                        heightbbox = float(label_info[4]) * imageHeight
                    
                        if label_info[0] == '0':
                            pc2cmbbox = width/widthbbox
                            height = heightbbox * pc2cmbbox
                        
                else:
                    angle = calculate_angle(imageCenital, imageBBOX, yawDegreesCenital)
        

                
                report_dict[IDAntena] = {
                        
                "clase"    : int(label_info[0]),
                "modelo"   : modeloTipo[int(label_info[0])],
                "high"     : height,
                "width"    : width,
                "angle"    : angle,
                "filename" : label_path.split('/')[-1].split('.')[0]
                
                }
                

            return IDAntena, report_dict
        
# Funcion que saca infromacion de label y rellena el reporte 
def get_high(label_path, report_dict, IDAntena, imageFrontal, alturaTorre):    
    import cv2
    print(label_path)
    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            labels = file.readlines()
        if len(labels) == 0:
            return IDAntena, report_dict
        else:
            for label in labels:
                IDAntena += 1
                # Formato YOLO: <class_id> <x_center> <y_center> <width> <height>
                label_info = label.split()
                if True:
                    
                    imageFrontalData = cv2.imread(imageFrontal)
                    imageBBOX = drawbbox(imageFrontalData, label_info,  0)
                    highPoint = hightTower(imageBBOX)
                    alturaCmAntena = int(input("Ingrese la altura de la antena en cm:"))
                    anchoCmAntena = int(input("Ingrese la ancho de la antena en cm:"))
                    px2cm, puntoMedio = pointAntena(imageFrontalData, alturaCmAntena)
                    
                    dist = np.linalg.norm(np.array(puntoMedio) - np.array(highPoint))
                    distCm = dist * px2cm
                    posAltura = int(alturaTorre) - int(distCm)
                    
                    # Agregar "alturaTorre" a report_dict sin sobreescribir otros valores
                    dictAntena = report_dict[str(IDAntena)]
                    dictAntena["alturaTorre"] = posAltura
                    dictAntena["width"] = anchoCmAntena
                    dictAntena["high"] = alturaCmAntena
                    report_dict[IDAntena] = dictAntena
                
                

                
                    print(f"report_dict {IDAntena}: {report_dict[IDAntena]}")
                
                
                
                
            return IDAntena, report_dict
    
def detectImg(imgPath, labelPath, cropPath, IDantena):
    # Read the first image
    if os.stat(labelPath).st_size == 0:
            return IDantena
    imgOR = cv2.imread(imgPath)
    imgC = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)
    
    img_height, img_width = img1.shape
    bbox = []
    
    # Read the corresponding label file
    with open(labelPath, 'r') as file:
        
        for line in file:
            line = line.split()
            _ , xc, yc, bw, bh = map(float, line)
            
            # Calculate bounding box coordinates
            x = int((xc - bw / 2) * img_width)
            y = int((yc - bh / 2) * img_height)
            x2 = int((xc + bw / 2) * img_width)
            y2 = int((yc + bh / 2) * img_height)
            
            # Ensure the bounding box is within image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            bbox.append([x, y, x2, y2])
    
    margen = 25
    for x, y, x2, y2 in bbox:
        IDantena += 1
        if x < x2 and y < y2:  # Check for valid bounding box
            x_margen = max(x - margen, 0)
            y_margen = max(y - margen, 0)
            x2_margen = min(x2 + margen, img1.shape[1])
            y2_margen = min(y2 + margen, img1.shape[0])

            # Realizar el corte con el margen añadido
            detection = imgOR[y_margen:y2_margen, x_margen:x2_margen]

            if detection.size > 0:
                cv2.imwrite(f'{cropPath}/{IDantena}.JPG', detection)
                
    return IDantena          
                
       
               
                