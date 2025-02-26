# Proyectos-SGBA1

## Estructura de ficheros 

root/
├── data/
│   ├── raw/               # Datos sin procesar (originales)
│   └── processed/         # Datos procesados/listos para el modelo
├── src/                   # Código fuente del proyecto
│   ├── data/              # Scripts para cargar y procesar datos
│   │   └── make_dataset.py
│   └── modelos/           # Scripts y notebooks para entrenar y evaluar modelos
│       ├── predictivo_precio/
│       │   ├── notebooks/
│       │   └── scripts/
│       ├── predictivo_hogar/
│       └── agente/
├── entorno/               # Modelos entrenados y checkpoints
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Descripción general del proyecto