# Guía Básica de Uso de DVC para el Versionado de Datos

## Introducción
DVC (Data Version Control) es una herramienta de control de versiones para datos y modelos de machine learning. Permite gestionar grandes volúmenes de datos de manera eficiente y reproducible. En esta guía, también veremos cómo integrar DVC con DagsHub para un flujo de trabajo más robusto.

## Instalación
Para instalar DVC, puedes usar `pip`:
```bash
# dvc-s3 es necesario porque bamos a conectarnos remotamente a un s3
pip install dvc dvc-s3
```

## Inicialización de un Proyecto DVC
Para inicializar un proyecto con DVC, navega al directorio de tu proyecto y ejecuta:
```bash
dvc init
```
Esto creeará en el directorio una carpeta .dvc

## Añadir Datos al Control de Versiones
Para añadir un archivo o directorio al control de versiones de DVC, usa el comando `dvc add`:
```bash
dvc add path/to/data
```
Esto creará un archivo `.dvc` que rastrea el archivo o directorio especificado.

## Almacenar Datos Remotamente
Configura un almacenamiento remoto para tus datos. Aquí usaremos DagsHub:
```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/auditoria.SGBA1/Proyectos-SGBA1.s3
```
Luego, modifica las credenciales de acceso de manera local:
```bash
dvc remote modify origin --local access_key_id XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
dvc remote modify origin --local secret_access_key XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
Sube los datos al almacenamiento remoto:
```bash
dvc push
```

## Recuperar Datos
Para recuperar datos desde el almacenamiento remoto, usa:
```bash
dvc pull
```

## Seguimiento de Cambios
Para rastrear cambios en tus datos, usa `dvc status`:
```bash
dvc status
```

## Ejemplo Completo
1. Inicializa DVC:
    ```bash
    dvc init
    ```
2. Añade datos:
    ```bash
    dvc add data/mydataset
    ```
3. Configura el almacenamiento remoto:
    ```bash
    dvc remote add -d myremote s3://mybucket/path
    ```
4. Modifica las credenciales de acceso:
    ```bash
    dvc remote modify origin --local access_key_id XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    dvc remote modify origin --local secret_access_key XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    ```
5. Sube los datos:
    ```bash
    dvc push
    ```
6. Recupera los datos en otro entorno:
    ```bash
    dvc pull
    ```

