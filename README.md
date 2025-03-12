# Proyectos-SGBA1

## Estructura de ficheros 

```
root/
├── .dvc
├── data/                   # Almacén de datos
│   ├── data_cleaning/      # Código para procesar los datos
│   │   ├── datos_clima
│   │   ├── datos_consumo
│   │   └── datos_precio
│   ├── processed/          # Datos procesados/listos para el modelo
│   │   ├── datos_clima
│   │   ├── datos_consumo
│   │   └── datos_precio
│   └── raw/                # Datos sin procesar (originales)
│       ├── datos_clima
│       │   ├── "idema1"
│       │   ├── "idema2"
│       │   └── "idema3"
│       ├── datos_consumo
│       └── datos_precio
├── guias
│   ├── entorno               
│   └── dvc
├── entorno/                # Configuración de entorno, manuales de instalación, modelos entrenados y checkpoints
├── src/                    # Código fuente del proyecto
│   ├── data/               # Scripts para cargar y procesar datos
│   │   └── make_dataset.py
│   └── modelos/            # Scripts y notebooks para entrenar y evaluar modelos
│       ├── agente/
│       │   ├── notebooks/
│       │   └── scripts/
│       ├── predictivo_hogar/
│       │   ├── notebooks/
│       │   └── scripts/
│       └── predictivo_precio/
│           ├── notebooks/
│           └── scripts/
├── pruebas/                # Espacio de experimentación y pruebas
│   ├── Modelos
│   ├── Prefect
│   ├── Prototipos
│   └── ml_env_v2
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Descripción general de la estructura del proyecto
```
