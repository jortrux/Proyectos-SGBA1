import os
import glob
import pandas as pd

carpeta = "downloads_generacion_consumo"  # Asegúrate de que esta ruta es correcta
archivos_csv = glob.glob(os.path.join(carpeta, "*.csv"))

print("Archivos encontrados:", archivos_csv)

if not archivos_csv:
    print("No se encontraron archivos CSV en la carpeta.")
    exit()

lista_df = []

for archivo in archivos_csv:
    df = pd.read_csv(archivo, delimiter=';')
    lista_df.append(df)

df_concatenado = pd.concat(lista_df, ignore_index=True)

salida = os.path.join(os.getcwd(), "Generacion_Consumo.csv")
df_concatenado.to_csv(salida, index=False, sep=';')

print(f"Se han combinado {len(archivos_csv)} archivos en {salida}")
