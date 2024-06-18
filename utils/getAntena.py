import cv2
from tqdm import tqdm
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.feature import hog
import shutil


def read_metadata(file_path):
    with open(file_path, 'r') as file:
        metadata = json.load(file)
    return metadata

def detectImg(imgPath, labelPath):
    # Read the first image
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
    
    # cortar detecciones de la imagen y darles mas espacio alrededor de la antena
    detections = []
    i = 0
    margen = 10
    target_size = (128, 128)
    for x, y, x2, y2 in bbox:
        if x < x2 and y < y2:  # Check for valid bounding box
            x_margen = max(x - margen, 0)
            y_margen = max(y - margen, 0)
            x2_margen = min(x2 + margen, img1.shape[1])
            y2_margen = min(y2 + margen, img1.shape[0])

            # Realizar el corte con el margen añadido
            detection = img1[y_margen:y2_margen, x_margen:x2_margen]

            if detection.size > 0:
                resized_detection = cv2.resize(detection, target_size)
                # Aplicar filtro Gaussiano y guardar en detections2
                filtered_detection = cv2.GaussianBlur(resized_detection, (5, 5), 0)
                detections.append(filtered_detection)                    
                imgName = os.path.basename(imgPath)
                # cv2.imwrite(f'crop/{imgName}_{i}.JPG', resized_detection)
                i += 1
    return detections

def detectImgFilter(imgPath, labelPath):
    # Read the first image
    imgC = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)
    
    img_height, img_width = img1.shape
    bbox = []
    
    # Read the corresponding label file
    with open(labelPath, 'r') as file:
        for line in file:
            line = line.split()
            _, xc, yc, bw, bh = map(float, line)
            
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
    
    # Cortar detecciones de la imagen y darles más espacio alrededor de la antena
    detections = []
    i = 0
    margen = 10
    target_size = (128, 128)
    for x, y, x2, y2 in bbox:
        if x < x2 and y < y2:  # Check for valid bounding box
            x_margen = max(x - margen, 0)
            y_margen = max(y - margen, 0)
            x2_margen = min(x2 + margen, img1.shape[1])
            y2_margen = min(y2 + margen, img1.shape[0])

            # Realizar el corte con el margen añadido
            detection = img1[y_margen:y2_margen, x_margen:x2_margen]

            if detection.size > 0:
                resized_detection = cv2.resize(detection, target_size)
                imgName = os.path.basename(imgPath)
                     
                
                # Aplicar filtro de Sobel
                sobelx = cv2.Sobel(resized_detection, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(resized_detection, cv2.CV_64F, 0, 1, ksize=5)
                sobel_combined = cv2.magnitude(sobelx, sobely)
                
                # Convertir la salida de Sobel a CV_8U
                sobel_combined = cv2.convertScaleAbs(sobel_combined)
                detections.append(sobel_combined)
                # cv2.imwrite(f'crop/{imgName}_{i}-Filter.JPG', sobel_combined)
                
                i += 1
    return detections


def foundBBOX(labelPath, idx):
    with open(labelPath, 'r') as file:
        for i, line in enumerate(file):
            if i == idx:
                line = line.split()
                _ , xc, yc, bw, bh = map(float, line)
                return xc, yc, bw, bh

def getFeatures(antena, extractores):
    for extractor in extractores:
        _, desc = extractor.detectAndCompute(antena, None)
        if desc is not None:
            desc = desc.astype(np.float32)
        return desc
        
def getLBPFeatures(antena, numPoints=24, radius=8):
    lbp = local_binary_pattern(antena, numPoints, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist  

def getHOGFeatures(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    hog_features = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations, block_norm='L2-Hys', visualize=False)
    return hog_features

def get_distPIX(img1, img2, bf):
    imgD1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    imgData1 = cv2.cvtColor(imgD1, cv2.COLOR_RGB2GRAY)
    
    imgD2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)
    imgData2 = cv2.cvtColor(imgD2, cv2.COLOR_RGB2GRAY)
    
    # Crear el detector SIFT
    SIFT = cv2.SIFT_create()
    print("Detectando puntos clave y descriptores para encontrar Distancia Pixeles...")
    kp1, des1 = SIFT.detectAndCompute(imgData1, None)
    kp2, des2 = SIFT.detectAndCompute(imgData2, None)
    
    # sacar cordenada del centro de la imagen
    x1c = int(imgData1.shape[1] / 2)
    y1c = int(imgData1.shape[0] / 2)
    
    # Crear rectangulo del alto de la imagen y ancho de 1/3 de la imagen
    x_min = x1c - int(imgData1.shape[1] / 6)
    x_max = x1c + int(imgData1.shape[1] / 6)
    y_min = y1c - int(imgData1.shape[0] / 2)
    y_max = y1c + int(imgData1.shape[0] / 2)
    
    cv2.rectangle(imgD1, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.imwrite('center.jpg', imgD1)
    
    def is_within_center(pt):
        x, y = pt
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    # Filtrar puntos clave que estén dentro del área central
    kp1_center = [kp for kp in kp1 if is_within_center(kp.pt)]
    des1_center = [des1[idx] for idx, kp in enumerate(kp1) if is_within_center(kp.pt)]
    kp2_center = [kp for kp in kp2 if is_within_center(kp.pt)]
    des2_center = [des2[idx] for idx, kp in enumerate(kp2) if is_within_center(kp.pt)]
    
    if not des1_center or not des2_center:
        print("No se encontraron puntos clave en el área central de una o ambas imágenes.")
        return None
    
    des1_center = np.array(des1_center)
    des2_center = np.array(des2_center)
    
    # Usar el emparejador de fuerza bruta
    matches = bf.knnMatch(des1_center, des2_center, k=2)
    
    good_match = []
    for m, n in tqdm(matches, total=len(matches), desc='Getting Dist PIX'):
        if m.distance < 0.7 * n.distance:
            good_match.append(m)
    
    good_match = sorted(good_match, key=lambda x: x.distance)
    print(f"Encontrados {len(good_match)} buenos emparejamientos.")
    
    if len(good_match) == 0:
        print("No se encontraron buenos emparejamientos.")
        return None
        
    good_match = good_match[:10]
    img_matches = cv2.drawMatches(imgD1, kp1_center, imgD2, kp2_center, good_match, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Guardar la imagen con los emparejamientos
    cv2.imwrite('matches.jpg', img_matches)
    
    # Calcular la distancia en y entre los puntos clave
    distances = []
    for match in good_match:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1_center[img1_idx].pt
        (x2, y2) = kp2_center[img2_idx].pt
        # Normalizar puntos
        x1 = x1 / imgData1.shape[1]
        x2 = x2 / imgData2.shape[1]
        y1 = y1 / imgData1.shape[0]
        y2 = y2 / imgData2.shape[0]

        # Calcular distancia en eje y
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(distance)
    
    # Aproximar cuánto se movió la imagen en píxeles
    avg_distance = np.mean(distances)
    avg_distance = round(avg_distance, 2)
    
    return avg_distance

def get_distPIXPoint(img1, img2):
    imgD1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2RGB)
    imgD2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2RGB)

    # Función para recoger puntos de la imagen
    def select_points(image):
        points = []

        fig, ax = plt.subplots()
        ax.imshow(image)

        def onclick(event):
            if len(points) < 2:
                ix, iy = event.xdata, event.ydata
                points.append((ix, iy))
                ax.plot(ix, iy, 'ro')
                fig.canvas.draw()
            if len(points) == 2:
                plt.close(fig)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return points

    # Seleccionar puntos en la primera imagen
    ptosim1 = select_points(imgD1)
    # Seleccionar puntos en la segunda imagen
    ptosim2 = select_points(imgD2)

    # Calcular la distancia entre los puntos correspondientes
    distances = []
    for i in range(len(ptosim1)):
        (x1, y1) = ptosim1[i]
        (x2, y2) = ptosim2[i]
        
        # Normalizar puntos
        x1 = x1 / imgD1.shape[1]
        x2 = x2 / imgD2.shape[1]
        y1 = y1 / imgD1.shape[0]
        y2 = y2 / imgD2.shape[0]

        # Calcular distancia en eje y
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(distance)
    
    # Aproximar cuánto se movió la imagen en píxeles
    avg_distance = np.mean(distances)

    avg_distance = round(avg_distance, 2)
    
    return avg_distance

def get_distMetadata(img1, img2): 
    # abrir metadata de las imagenes
    metadata1 = read_metadata(img1)
    metadata2 = read_metadata(img2)
    
    FOV = float(metadata1["FOV"].split()[0])  # asumiendo que FOV es el mismo para ambas imágenes
    width1 = int(metadata1["ImageWidth"])
    height1 = int(metadata1["ImageHeight"])
    yaw1 = float(metadata1["GimbalYawDegree"])
    pitch1 = float(metadata1["GimbalPitchDegree"])

    width2 = int(metadata2["ImageWidth"])
    height2 = int(metadata2["ImageHeight"])
    yaw2 = float(metadata2["GimbalYawDegree"])
    pitch2 = float(metadata2["GimbalPitchDegree"])

     # Calcular diferencias angulares
    delta_yaw = abs(yaw1 - yaw2)


    # Convertir FOV a radianes
    FOV = np.radians(FOV)

    # Calcular la distancia en píxeles basada en la diferencia de ángulo y el FOV
    pixel_distance_x = (delta_yaw / FOV) 


    # Calcular distancia euclidiana normalizada en píxeles
    pixel_distance_total = pixel_distance_x

    # Normalizar la distancia total por el tamaño de la imagen
    normalized_distance = pixel_distance_total * 10/ width1

    return round(normalized_distance, 5)
       
    


def get_bbox(labelPath):
       
    bbox = []
    
    # Read the corresponding label file
    with open(labelPath, 'r') as file:
        for line in file:
            line = line.split()
            _ , xc, yc, bw, bh = map(float, line)
            
            
            bbox.append([xc, yc, bw, bh])
    
    return bbox

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 1.0

def getScores(matches):
    match_counts = {key: len(matches) for key, matches in matches.items()}
    # Encontrar el máximo y mínimo número de good matches
    max_matches = max(match_counts.values())
    min_matches = min(match_counts.values())
    # Normalizar los valores
    return {key: normalize(count, min_matches, max_matches) for key, count in match_counts.items()}

def euclidean_distance(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def getScoreLH(distances):
    
    max_distance = max(distances.values())
    min_distance = min(distances.values())

    # Normaliza las distancias para obtener puntuaciones entre 0 y 1
    scores = {key: 1 - (dist - min_distance) / (max_distance - min_distance) if max_distance != min_distance else 1.0
            for key, dist in distances.items()}

    return scores

def registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2, notPair = False):
    if len(IDAntenas) == 0:
        print("No hay antenas en la lista")
        
        if not notPair:
            IDAntenas.append([IDAntena, best_antenna])
        if notPair:
            IDAntenas.append([IDAntena, None])
        realID = 0
        print(f"Antena {IDAntena} asignada como {realID}")
        
        xc1, yc1, bw1, bh1 = foundBBOX(os.path.join(labelsPath, images[IDimg1].replace('.JPG', '.txt')), IDAntena)
        x1 = int((xc1 - bw1 / 2) * img1.shape[1])
        y1 = int((yc1 - bh1 / 2) * img1.shape[0])
        x2 = int((xc1 + bw1 / 2) * img1.shape[1])
        y2 = int((yc1 + bh1 / 2) * img1.shape[0])
        cv2.circle(img1, (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), 5, (0,0,250), -1)
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0,0,250), 10)
        cv2.putText(img1, str(realID), (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        
        if not notPair:
            xc2, yc2, bw2, bh2 = foundBBOX(os.path.join(labelsPath, images[IDimg2].replace('.JPG', '.txt')), best_antenna)
            x1 = int((xc2 - bw2 / 2) * img2.shape[1])
            y1 = int((yc2 - bh2 / 2) * img2.shape[0])
            x2 = int((xc2 + bw2 / 2) * img2.shape[1])
            y2 = int((yc2 + bh2 / 2) * img2.shape[0])
            cv2.circle(img2, (int(xc2 * img2.shape[1]), int(yc2 * img2.shape[0])), 5, (0,0,250), -1)
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0,0,250), 10)
            cv2.putText(img2, str(realID), (int(xc2 * img2.shape[1]), int(yc2 * img2.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        
    
    
    elif not notPair:
        found = False
        realID = None
        for i in range(len(IDAntenas)):
                                                
            # Cambia de img  
            if len(IDAntenas[i]) >= IDimg1:
                if IDAntenas[i][IDimg1] == IDAntena:
                    print(f"Antena {IDAntena} ya registrada como {i}")
                    IDAntenas[i].append(best_antenna)
                    found = True
                    realID = i 

                    xc2, yc2, bw2, bh2 = foundBBOX(os.path.join(labelsPath, images[IDimg2].replace('.JPG', '.txt')), best_antenna)
                    x1 = int((xc2 - bw2 / 2) * img2.shape[1])
                    y1 = int((yc2 - bh2 / 2) * img2.shape[0])
                    x2 = int((xc2 + bw2 / 2) * img2.shape[1])
                    y2 = int((yc2 + bh2 / 2) * img2.shape[0])
                    cv2.circle(img2, (int(xc2 * img2.shape[1]), int(yc2 * img2.shape[0])), 5, (0,0,250), -1)
                    cv2.rectangle(img2, (x1, y1), (x2, y2), (0,0,250), 10)
                    cv2.putText(img2, str(realID), (int(xc2 * img2.shape[1]), int(yc2 * img2.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                
                                                
        if not found:
            print(f"Antena {IDAntena} no registrada")
            FoundAntenas = []
            for i in range(IDimg1+1):
                FoundAntenas.append(None)
                            
            FoundAntenas[IDimg1] = IDAntena
            FoundAntenas.append(best_antenna)
            IDAntenas.append(FoundAntenas)
            realID = len(IDAntenas) - 1
            print(f"Asignando como Antena {realID}")

            xc1, yc1, bw1, bh1 = foundBBOX(os.path.join(labelsPath, images[IDimg1].replace('.JPG', '.txt')), IDAntena)
            x1 = int((xc1 - bw1 / 2) * img1.shape[1])
            y1 = int((yc1 - bh1 / 2) * img1.shape[0])
            x2 = int((xc1 + bw1 / 2) * img1.shape[1])
            y2 = int((yc1 + bh1 / 2) * img1.shape[0])
            cv2.circle(img1, (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), 5, (0,0,250), -1)
            cv2.rectangle(img1, (x1, y1), (x2, y2), (0,0,250), 10)
            cv2.putText(img1, str(realID), (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            
            xc2, yc2, bw2, bh2 = foundBBOX(os.path.join(labelsPath, images[IDimg2].replace('.JPG', '.txt')), best_antenna)
            x1 = int((xc2 - bw2 / 2) * img2.shape[1])
            y1 = int((yc2 - bh2 / 2) * img2.shape[0])
            x2 = int((xc2 + bw2 / 2) * img2.shape[1])
            y2 = int((yc2 + bh2 / 2) * img2.shape[0])
            cv2.circle(img2, (int(xc2 * img2.shape[1]), int(yc2 * img2.shape[0])), 5, (0,0,250), -1)
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0,0,250), 10)
            cv2.putText(img2, str(realID), (int(xc2 * img2.shape[1]), int(yc2 * img2.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        
    elif notPair:
        print("No se encontro par en IMG2, se asigna antena solo a IMG1")
        found = False
        realID = None
        for i in range(len(IDAntenas)):                     
            if len(IDAntenas[i]) == IDimg1 + 1:
                if IDAntenas[i][IDimg1] == IDAntena:
                    print(f"Antena {IDAntena} ya registrada como {i}")
                    IDAntenas[i].append(None)
                    found = True
                    realID = i 
                    xc1, yc1, bw1, bh1 = foundBBOX(os.path.join(labelsPath, images[IDimg1].replace('.JPG', '.txt')), IDAntena)
                    x1 = int((xc1 - bw1 / 2) * img1.shape[1])
                    y1 = int((yc1 - bh1 / 2) * img1.shape[0])
                    x2 = int((xc1 + bw1 / 2) * img1.shape[1])
                    y2 = int((yc1 + bh1 / 2) * img1.shape[0])
                    cv2.circle(img1, (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), 5, (0,0,250), -1)
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0,0,250), 10)
                    cv2.putText(img1, str(realID), (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                    
                    
        if not found:
            print(f"Antena {IDAntena} no registrada y no tiene par en IMG2")
            FoundAntenas = []
            for i in range(IDimg1+1):
                FoundAntenas.append(None)
                            
            FoundAntenas[IDimg1] = IDAntena
            FoundAntenas.append(best_antenna)
            IDAntenas.append(FoundAntenas)
            realID = len(IDAntenas) - 1
            print(f"Asignando como Antena {realID}")

            xc1, yc1, bw1, bh1 = foundBBOX(os.path.join(labelsPath, images[IDimg1].replace('.JPG', '.txt')), IDAntena)
            x1 = int((xc1 - bw1 / 2) * img1.shape[1])
            y1 = int((yc1 - bh1 / 2) * img1.shape[0])
            x2 = int((xc1 + bw1 / 2) * img1.shape[1])
            y2 = int((yc1 + bh1 / 2) * img1.shape[0])
            cv2.circle(img1, (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), 5, (0,0,250), -1)
            cv2.rectangle(img1, (x1, y1), (x2, y2), (0,0,250), 10)
            cv2.putText(img1, str(realID), (int(xc1 * img1.shape[1]), int(yc1 * img1.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
                

    return IDAntenas, img1, img2
           

def getAntenaIDS(folder):

    extractores = [cv2.SIFT_create()]
    # Lista de 50 colores distintos
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(50)]
    
    IDAntenas = []
    labelsPath = os.path.join(folder, 'labels')
    os.makedirs(labelsPath, exist_ok=True)
    imagesPath = os.path.join(folder, 'images')
    images = os.listdir(imagesPath)
    for file in images:
        if file.endswith('.txt'):
            # Mover el archivo a la carpeta labels shutil
            shutil.move(os.path.join(imagesPath, file), os.path.join(labelsPath, file))
    images = os.listdir(imagesPath)        
    from utils.metadata import get_metadata
    print("folder", folder)
    get_metadata([folder])           
    labelsPath = os.path.join(folder, 'labels')
    labels = os.listdir(labelsPath)
    metadataPath = os.path.join(folder, 'metadata')
    metadatas = os.listdir(metadataPath)
    os.makedirs('crop', exist_ok=True) 
    antenasDetected = [detectImg(os.path.join(imagesPath, image), os.path.join(labelsPath, image.replace('.JPG', '.txt'))) for image in tqdm(images, desc='Detecting antennas')]
    antenasDetectedFilter = [detectImgFilter(os.path.join(imagesPath, image), os.path.join(labelsPath, image.replace('.JPG', '.txt')) ) for image in tqdm(images, desc='Detecting antennas with Gaussian filter')]
    readImg = [cv2.imread(os.path.join(imagesPath, image)) for image in tqdm(images, desc='Reading images')]
    bf = cv2.BFMatcher()
    
    numImages = len(images)

    for IDimg1 in range(numImages-1):      
        IDimg2 = IDimg1 + 1            
        print(f'Processing images {IDimg1} and {IDimg2}')

        allScores = {}
                

        img1 = readImg[IDimg1]
        
        img2 = readImg[IDimg2]
        
        antenaSelected = []
        
        distPIX = get_distMetadata(os.path.join(metadataPath, metadatas[IDimg1]), os.path.join(metadataPath, metadatas[IDimg2]))
        # distPIX = get_distPIXPoint(img1Path, img2Path)
        print(f'Distancia en pixeles: {distPIX}')

                
        for IDAntena, antena1 in tqdm(enumerate(antenasDetected[IDimg1]), total=len(antenasDetected[IDimg1]), desc='Finding matches'):
            xc1, yc1, bw1, bh1 = foundBBOX(os.path.join(labelsPath, images[IDimg1].replace('.JPG', '.txt')), IDAntena)

            antena1Filter = antenasDetectedFilter[IDimg1][IDAntena]
            
            descs = getFeatures(antena1, extractores)
            descsFilter = getFeatures(antena1Filter, extractores)
            hogsFilter = getHOGFeatures(antena1Filter)
            
            
            siftMatches = {}
            lbpDist = {}
            hogDist = {}
            pixDist = {}
            siftMatchesFilter = {}
            lbpDistFilter = {}
            hogDistFilter = {}
            
            if len(antenasDetected[IDimg2]) != 0:
                for i, antena in enumerate(antenasDetected[IDimg2]):
                    xc2, yc2, bw2, bh2 = foundBBOX(os.path.join(labelsPath, images[IDimg2].replace('.JPG', '.txt')), i)
                    # Calcular distancia en pixeles entre xc1, yc1 y xc2, yc2
                    dist = np.sqrt((xc2 - xc1) ** 2 + (yc2 - yc1) ** 2)
                    dist = round(dist, 3)

                    # Calcular que tan cerca esta dist de distPIX
                    prom = np.exp(-1 * np.abs((dist - distPIX) / distPIX))
                    prom = round(prom, 3)    
                    pixDist[i] = prom    
                    
                    descs2 = getFeatures(antena, extractores)
                    matches = bf.knnMatch(descs, descs2, k=2)
                    good_match = []
                    for m in matches:
                        if m[0].distance/m[1].distance < 0.7:
                            good_match.append(m)
                    good_match_arr = np.asarray(good_match)
                    siftMatches[i] = good_match_arr
                    
                    descs2Filter = getFeatures(antenasDetectedFilter[IDimg2][i], extractores)
                    matchesFilter = bf.knnMatch(descsFilter, descs2Filter, k=2)
                    good_matchFilter = []
                    for m in matchesFilter:
                        if m[0].distance/m[1].distance < 0.7:
                            good_matchFilter.append(m)
                    good_match_arrFilter = np.asarray(good_matchFilter)
                    siftMatchesFilter[i] = good_match_arrFilter
                                        
                    
                    hogs2Filter = getHOGFeatures(antenasDetectedFilter[IDimg2][i])
                    hogDistFilter[i] = euclidean_distance(hogsFilter, hogs2Filter)

                    
                                
            
                hogScore = getScoreLH(hogDistFilter)
                siftScore = getScores(siftMatches)
                siftScoreFilter = getScores(siftMatchesFilter)    

                scores = {}
                
                weight_pix = 0.4
                weight_sift = 0.1
                weight_siftFilter = 0.25
                weight_hog = 0.25

                for i in range(len(antenasDetected[IDimg2])):
                    scores[i] = weight_pix * pixDist[i] + weight_sift * siftScore[i] + weight_siftFilter * siftScoreFilter[i] + weight_hog * hogScore[i]
                            
                allScores[IDAntena] = scores   
                
            best_score = {}
            if len(antenasDetected[IDimg2]) > 0:
                for IDAntena in allScores:
                    # guardar [key, value] de la antena con mejor score
                    best_score[IDAntena] = [max(allScores[IDAntena], key=allScores[IDAntena].get), max(allScores[IDAntena].values())]        
                
                repetidos = True
                while repetidos:
                    for IDAntena in best_score:
                        # encontrar values repetidos
                        if best_score[IDAntena][0] is not None:
                            keysRep  = [key for key, value in best_score.items() if value[0] == best_score[IDAntena][0]]
                            if len(keysRep) > 1:
                                # encontrar el key con mejor score best_score[key][1]
                                scoreRep = []
                                for key in keysRep:
                                    scoreRep.append(best_score[key][1])
                                
                                maxScore = max(scoreRep)
                                idxMaxScore = scoreRep.index(maxScore)
                                # print(f"Antena {keysRep[idxMaxScore]} is the best antenna with score {maxScore}")
                                keysRep.pop(idxMaxScore)
                                for key in keysRep:
                                    oldScore = best_score[key][0]
                                    allScores[key].pop(oldScore)
                                    if len(allScores[key]) > 0:
                                        best_score[key] = [max(allScores[key], key=allScores[key].get), max(allScores[key].values())]
                                    else:
                                        best_score[key] = [None, 0]

                
                    noRepetidos = 0
                    for IDAntena in best_score:
                        keysRep  = [key for key, value in best_score.items() if value[0] == best_score[IDAntena][0]]
                        if len(keysRep) == 1:
                            noRepetidos += 1
                            
                        elif best_score[IDAntena][0] is None:
                            noRepetidos += 1
                            
                    if noRepetidos == len(best_score):
                        repetidos = False
                            
                for IDAntena in best_score:
                    best_antenna = best_score[IDAntena][0]
                    best_nota = best_score[IDAntena][1] 
                                    
                    if best_nota > 0.65:
                        print(f"Finding {IDAntena}...")
                        print(f"Best antenna: {best_antenna} with score {best_nota}")
                        
                        IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2)
                        readImg[IDimg1] = img1
                        readImg[IDimg2] = img2
                        
                    elif IDimg1 == 0:
                        print(f"Antena {IDAntena} con nota muy baja y es la primera imagen")
                        IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2, notPair = True)
                        readImg[IDimg1] = img1                                        

                    elif IDimg1 != 0 and best_antenna != None:
                            print(f"Antena {IDAntena} con nota muy baja")                     
                            print(f"Best antenna: {best_antenna} with score {best_nota}")
                            IDimg0 = IDimg1 - 1
                            IDGlobal = None
                            for e, antenas in enumerate(IDAntenas):
                                
                                if antenas[IDimg1] == IDAntena:
                                    IDAntena0 = antenas[IDimg0]
                                    IDGlobal = e
                                    print(f"Antena {IDAntena} ya registrada como {IDAntena0} de imagen {IDimg0} y La ID Global es {IDGlobal}")
                                
                                    
                                    distNew = get_distMetadata(os.path.join(metadataPath, metadatas[IDimg0]), os.path.join(metadataPath, metadatas[IDimg2]))
                                    xc0, yc0, bw0, bh0 = foundBBOX(os.path.join(labelsPath, images[IDimg0].replace('.JPG', '.txt')), IDAntena0)
                                    xc2, yc2, bw2, bh2 = foundBBOX(os.path.join(labelsPath, images[IDimg2].replace('.JPG', '.txt')), best_antenna)
                                    
                                    dist = np.sqrt((xc2 - xc0) ** 2 + (yc2 - yc0) ** 2)
                                    dist = round(dist, 3)
                                    print(f"Distancia entre antenas {IDimg0} - {IDimg2}: {dist}")
                                    # Calcular que tan cerca esta dist de distPIX
                                    prom = np.exp(-1 * np.abs((dist - distPIX) / distPIX))
                                    prom = round(prom, 3)    
                                    print(f"Promedio de distancia entre antenas {IDimg0} - {IDimg2}: {prom}")     
                                                    
                                    antena0 = antenasDetected[IDimg0][IDAntena0]
                                    antena0Filter = antenasDetectedFilter[IDimg0][IDAntena0]
                                    descs = getFeatures(antena0, extractores)
                                    descsFilter = getFeatures(antena0Filter, extractores)
                                    hogsFilter = getHOGFeatures(antena0Filter)
                                    
                                    antena2 = antenasDetected[IDimg2][best_antenna]
                                    descs2 = getFeatures(antena2, extractores)
                                    descs2Filter = getFeatures(antenasDetectedFilter[IDimg2][best_antenna], extractores)
                                    hogs2Filter = getHOGFeatures(antenasDetectedFilter[IDimg2][best_antenna])
                                    

                                    if prom > 0.45:
                                        print(f"Best antena de {IDAntena} muy cerca de la antena {IDAntena0} de la imagen anterior")
                                        IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2)
                                        readImg[IDimg1] = img1
                                        readImg[IDimg2] = img2                                   
                                
                                    else:
                                        print(f"Antena {IDAntena} nota baja, no se encontro antena cercana")
                                        IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2, notPair = True)
                                        readImg[IDimg1] = img1
                    for listAntenas in IDAntenas:
                        if len(listAntenas) < IDimg2 + 1:
                            listAntenas.append(None)            
                                
                            if IDGlobal is None:
                                print(f"Antena {IDAntena} no esta registrada en la imagen anterior")
                                if prom > 0.45:
                                    print(f"Best antenna: {best_antenna} with score {best_nota}")
                                    IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2)
                                    readImg[IDimg1] = img1
                                    readImg[IDimg2] = img2
                                    
                                else:
                                    IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2, notPair = True)
                                    readImg[IDimg1] = img1
                    else:
                        print(f"Antena {IDAntena} no cumple con nada")
                        IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2, notPair = True)
                        readImg[IDimg1] = img1
                                
          
                    antenaSelected.append(best_antenna)
                for listAntenas in IDAntenas:
                    if len(listAntenas) < IDimg2 + 1:
                        listAntenas.append(None)
                
            else:
                print(f"Imagen {IDimg2} no tiene antenas detectadas")                     
                IDAntenas, img1, img2 = registerAntena(IDAntenas, IDAntena, best_antenna, images, IDimg1, IDimg2, labelsPath, img1, img2, notPair = True)
                readImg[IDimg1] = img1
                for listAntenas in IDAntenas:
                    if len(listAntenas) < IDimg2 + 1:
                        listAntenas.append(None)
                
                
             
        for listAntenas in IDAntenas:
            if len(listAntenas) < IDimg2 + 1:
                listAntenas.append(None)
        print(f"IDAntenas: {IDAntenas}")  
        
    
        
    print("Antenas ya procesadas")       
    detectPath = os.path.join(folder, 'detections')
    os.makedirs(detectPath, exist_ok=True) 
    for i, img in tqdm(enumerate(readImg), total = len(readImg), desc='Saving images'):
        cv2.imwrite(os.path.join(detectPath,images[i]), img)    
            
            
    dictImages = {}
    for e, image in enumerate(images):
        dictImages[e] = image
                
    interfaz = True
    if interfaz:          
        from utils.interfazG import mainInterfaz
        newAntena = mainInterfaz(IDAntenas)
        finalAntenas = []
        for listAntenas in newAntena:
            allAntenas = []
            for idAntena in listAntenas:
                if allAntenas == []:
                    allAntenas = IDAntenas[idAntena]
                else:
                    for i in range(len(allAntenas)):
                        if allAntenas[i] == None:
                            allAntenas[i] = IDAntenas[idAntena][i]
            finalAntenas.append(allAntenas)
        print(f"Final Antenas: {finalAntenas}")

    print(f"Se detectaron {len(IDAntenas)} antenas.")
    print(f"IDAntenas: {IDAntenas}")
    
    report_dict = {}
    for IDAntena in range(len(finalAntenas)):
        report_dict[IDAntena] = {
            "high": None,
            "width": None,
            "model": "RRU",
            "filenames": list(),
        }
        for antena in finalAntenas[IDAntena]:
            report_dict[IDAntena]["filenames"].append(dictImages[antena])
            



        
    return IDAntenas, dictImages
            
