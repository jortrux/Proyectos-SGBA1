import os
import glob
import pandas as pd

# Ruta de la carpeta donde se encuentran los CSV
carpeta = "downloads_generacion_consumo"  # Asegúrate de que esta ruta es correcta

# Buscar todos los archivos CSV en la carpeta
archivos_csv = glob.glob(os.path.join(carpeta, "*.csv"))

print("Archivos encontrados:", archivos_csv)

if not archivos_csv:
    print("No se encontraron archivos CSV en la carpeta.")
    exit()

# Lista para almacenar cada DataFrame leído
lista_df = []

for archivo in archivos_csv:
    df = pd.read_csv(archivo, delimiter=';')
    lista_df.append(df)

# Unir todos los DataFrames en uno solo
df_concatenado = pd.concat(lista_df, ignore_index=True)

# Convertir la columna 'datetime' a tipo fecha y ordenar de más antiguo a más nuevo
df_concatenado['datetime'] = pd.to_datetime(df_concatenado['datetime'])
df_concatenado.sort_values(by='datetime', inplace=True)

# Guardar el DataFrame combinado y ordenado en el directorio actual
salida = os.path.join(os.getcwd(), "Generacion_Consumo.csv")
df_concatenado.to_csv(salida, index=False, sep=';')

print(f"Se han combinado {len(archivos_csv)} archivos en {salida}")
