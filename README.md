# Proyectos-SGBA1

## Estructura de ficheros 

```
root/
├── data/                   # Almacén de datos
│   ├── data_cleaning/      # Código para procesar los datos
│   ├── processed/          # Datos procesados/listos para el modelo
│   │   ├── agente/
│   │   ├── predictivo_hogar/
│   │   └── predictivo_precio/
│   └── raw/                # Datos sin procesar (originales)
│       ├── agente/
│       ├── predictivo_hogar/
│       └── predictivo_precio/
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
│   ├── ml_env_2/
│   ├── Modelos/
│   ├── Prefect/
│   └── Prototipos/
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Descripción general de la estructura del proyecto
```
