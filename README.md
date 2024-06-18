
### Reducción de Imágenes y Carga en AWS S3

#### Descripción
Este script está diseñado para interactuar con el sistema de gestión de tareas CVAT para obtener datasets específicos, modificar la calidad de las imágenes y cargarlas en un bucket de AWS S3. 

#### Pre-requisitos
- Python 3.6 o superior.
- Instalación de dependencias de Python especificadas en `requirements.txt` (ejemplo a continuación).
- Un archivo `.env` en el directorio raíz con las claves de AWS y detalles de conexión CVAT necesarios.
- Configuración adecuada en AWS para permitir la carga de archivos.

#### Instalación de dependencias
Para instalar las dependencias necesarias, ejecuta el siguiente comando en la terminal:
```
pip install -r requirements.txt
```

#### Archivo de configuración `.env`
Asegúrate de crear un archivo `.env` en el directorio raíz con la siguiente estructura:
```
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_DEFAULT_REGION=tu_region
AWS_BUCKET=nombre_de_tu_bucket
CVAT_HOST=host_de_cvat
CVAT_USERNAME=tu_usuario_cvat
CVAT_PASSWORD=tu_contraseña_cvat
```

#### Uso del script
Para usar el script debes seguir estos pasos:
1. **Crear task en Cvat:** Ingresar al sevidor Cvat (http://3.225.205.173:8080/), crear task en el proyecto Antenas Entel con el nombre `levID-medID` y seleccionar el CloudStorage las imágenes que corresponden a esa medición.
2. **Correr Script:** Ya con la task creada debes corrar el siguiente comando `python lowIMG.py levID-medID`.

Con esto las imágenes con menor calidad ya estarían en el S3 correspondiente a al levantamiento y medición.

#### Funciones principales del script
1. **Conexión con CVAT**: Establece una conexión con el servidor CVAT usando las credenciales proporcionadas.
2. **Descarga de datos**: Descarga un dataset de CVAT basado en el nombre de la tarea proporcionado.
3. **Procesamiento de imágenes**: Reduce la calidad de las imágenes descargadas y las almacena en una carpeta específica.
4. **Carga a AWS S3**: Sube las imágenes procesadas a un bucket de S3 especificado.
5. **Limpieza**: Elimina los archivos locales temporales creados durante el proceso.

#### Manejo de errores
El script maneja errores básicos relacionados con la autenticación de AWS y otros errores de ejecución, notificando al usuario sobre cualquier problema que impida la finalización del proceso.

#### Notas adicionales
- Asegúrate de tener permisos adecuados en AWS y CVAT para evitar problemas de acceso.
- Revisa los límites de almacenamiento en S3 y los ajustes de seguridad para evitar cargos inesperados o exposición de datos.

Cualquier duda contactarse con mgnavarrete.
