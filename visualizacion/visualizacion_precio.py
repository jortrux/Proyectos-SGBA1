import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import numpy as np

# Estilo visual mejorado
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

# Cargar datos
df_precio = pd.read_csv("prediccion_precio.csv", parse_dates=["timestamp"])
df_consumo = pd.read_csv("predicciones_consumo.csv", parse_dates=["fecha"])

# Renombrar columnas para consistencia
df_precio.rename(columns={"€/kwh": "precio_real", "prediccion_€/kwh": "precio_pred"}, inplace=True)
df_consumo.rename(columns={"real_consumo": "consumo_real", "pred_consumo": "consumo_pred"}, inplace=True)

# ----------- ANIMACIÓN PRECIO -----------

fig1, ax1 = plt.subplots()

# Mostrar siempre todos los datos reales desde el principio
line1_real, = ax1.plot(df_precio["timestamp"], df_precio["precio_real"], 
                      label="Precio real", linewidth=2, color='blue')

# La línea de predicción comienza vacía y se irá llenando
line1_pred, = ax1.plot([], [], label="Predicción", linestyle="--", 
                      linewidth=2, color='red')

# Configuración de los ejes y etiquetas
ax1.set_title("Precio de la luz (€/kWh)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Tiempo", fontsize=12)
ax1.set_ylabel("€/kWh", fontsize=12)

# Mejoras en la visualización del eje X
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
fig1.autofmt_xdate()  # Rota las etiquetas para mejor visualización

# Limitar ejes para mostrar todo el rango de datos
time_min = df_precio["timestamp"].min()
time_max = df_precio["timestamp"].max()
y_min = df_precio["precio_real"].min()
y_max = df_precio["precio_real"].max()
if not pd.isna(df_precio["precio_pred"]).all():
    y_min = min(y_min, df_precio["precio_pred"].min())
    y_max = max(y_max, df_precio["precio_pred"].max())

# Agregar margen para mejor visualización
y_margin = (y_max - y_min) * 0.1
ax1.set_xlim(time_min, time_max)
ax1.set_ylim(y_min - y_margin, y_max + y_margin)

ax1.legend(loc='best')
ax1.grid(True)

def init_precio():
    line1_pred.set_data([], [])
    return line1_pred,

def update_precio(frame):
    # Mostrar datos de predicción hasta el frame actual
    x = df_precio["timestamp"][:frame+1]
    y_pred = df_precio["precio_pred"][:frame+1]
    
    line1_pred.set_data(x, y_pred)
    return line1_pred,

# ----------- ANIMACIÓN CONSUMO -----------

fig2, ax2 = plt.subplots()

# Mostrar siempre todos los datos reales desde el principio
line2_real, = ax2.plot(df_consumo["fecha"], df_consumo["consumo_real"], 
                      label="Consumo real", linewidth=2, color='green')

# La línea de predicción comienza vacía y se irá llenando
line2_pred, = ax2.plot([], [], label="Predicción", linestyle="--", 
                      linewidth=2, color='purple')

# Configuración de los ejes y etiquetas
ax2.set_title("Consumo energético (kWh)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Tiempo", fontsize=12)
ax2.set_ylabel("kWh", fontsize=12)

# Mejoras en la visualización del eje X
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
fig2.autofmt_xdate()  # Rota las etiquetas para mejor visualización

# Limitar ejes para mostrar todo el rango de datos
time_min = df_consumo["fecha"].min()
time_max = df_consumo["fecha"].max()
y_min = df_consumo["consumo_real"].min()
y_max = df_consumo["consumo_real"].max()
if not pd.isna(df_consumo["consumo_pred"]).all():
    y_min = min(y_min, df_consumo["consumo_pred"].min())
    y_max = max(y_max, df_consumo["consumo_pred"].max())

# Agregar margen para mejor visualización
y_margin = (y_max - y_min) * 0.1
ax2.set_xlim(time_min, time_max)
ax2.set_ylim(y_min - y_margin, y_max + y_margin)

ax2.legend(loc='best')
ax2.grid(True)

def init_consumo():
    line2_pred.set_data([], [])
    return line2_pred,

def update_consumo(frame):
    # Mostrar datos de predicción hasta el frame actual
    x = df_consumo["fecha"][:frame+1]
    y_pred = df_consumo["consumo_pred"][:frame+1]
    
    line2_pred.set_data(x, y_pred)
    return line2_pred,

# Crear las animaciones
ani1 = FuncAnimation(fig1, update_precio, frames=len(df_precio), init_func=init_precio,
                     interval=500, blit=True, repeat=False)

ani2 = FuncAnimation(fig2, update_consumo, frames=len(df_consumo), init_func=init_consumo,
                     interval=500, blit=True, repeat=False)

# Ajustar el layout para evitar que se recorten las etiquetas
plt.tight_layout()
plt.ion()  # Modo interactivo
plt.show(block=False)