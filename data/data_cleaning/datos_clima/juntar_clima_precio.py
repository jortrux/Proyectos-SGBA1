import pandas as pd
import numpy as np
import os
import glob

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def rename_columns(df, idema):
    for col in df.columns:
        if col != 'fecha':
            # Renombramos las columnas para que no haya colisiones
            df.rename(columns={col: f"{col}_{idema}"}, inplace=True)
    return df

script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = script_dir.split('data/')[0]
os.chdir(working_dir) # Poner la carpeta data como directorio actual
print(f"Directorio actual: {os.getcwd()}")

# Por cada file en la carpeta de datos clima que termine en _clima_numerizado.parquet
# Lo cargamos y lo juntamos con el resto de ficheros

clima_files = glob.glob('data/processed/datos_clima/*_clima_numerizado.parquet')

df_clima = pd.DataFrame()

for file in clima_files:
    print(f"Procesando {file}")

    idema = file.split('/')[-1].split('_')[0]

    if df_clima.empty:
        df_clima = pd.read_parquet(file)
        df_clima = rename_columns(df_clima, idema)
    else:
        # Read the new file
        temp_df = pd.read_parquet(file)
        # Eliminar anio,mes,dia,dia_semana,estacion
        temp_df.drop(['anio', 'mes', 'dia', 'dia_semana', 'estacion'], axis=1, inplace=True)
        temp_df = rename_columns(temp_df, idema)
        # eliminar columnas que ya existen en el dataframe
        # anio,mes,dia,dia_semana,estacion


        # Merge with existing dataframe on the date column
        # Using outer join to keep all dates from both dataframes
        df_clima = pd.merge(df_clima, temp_df, on='fecha', how='outer')

# Ordenamos por fecha
df_clima.sort_values('fecha', inplace=True)
df_clima.reset_index(drop=True, inplace=True)

print(f"Guardando el dataframe con el clima de todas las estaciones")
df_clima.to_parquet('data/processed/datos_clima/clima_numerizado_all.parquet')
df_clima.to_csv('data/processed/datos_clima/clima_numerizado_all.csv', index=False)

# Procesar valores NaN
lista_estaciones = ['3196', '9434', '9170', '2539', '6155A', '5514', '1505', '3469A', '8019', '0200E', '4121', '8175']
# lista_estaciones = ['0200E']

for estacion in lista_estaciones:
    # Seleccionar columnas de esta estación
    columnas_numericas = df_clima.select_dtypes(include=[np.number]).columns
    cols_estacion = [col for col in columnas_numericas if f"_{estacion}" in col]

    # Contar NaN en las columnas de esta estación antes del procesamiento
    num_nan_before = df_clima[cols_estacion].isna().sum().sum()
    print(f"Estación {estacion}: {num_nan_before} valores NaN antes del procesamiento KNN")
        
    if cols_estacion:
        # Escalar
        scaler_estacion = StandardScaler()
        datos_estacion = df_clima[cols_estacion].fillna(df_clima[cols_estacion].mean())
        datos_escalados = scaler_estacion.fit_transform(datos_estacion)
        
        # Imputer
        imputer_estacion = KNNImputer(n_neighbors=7)  # Menos vecinos para datos específicos de estación
        datos_imputados = imputer_estacion.fit_transform(datos_escalados)
        
        # Desescalar
        datos_originales = scaler_estacion.inverse_transform(datos_imputados)
        
        # Actualizar DataFrame
        df_clima[cols_estacion] = datos_originales

        df_temp = pd.DataFrame(datos_originales, columns=cols_estacion)
        num_nan_after = df_temp.isna().sum().sum()
        print(f"Estación {estacion}: {num_nan_after} valores NaN después del procesamiento KNN")




