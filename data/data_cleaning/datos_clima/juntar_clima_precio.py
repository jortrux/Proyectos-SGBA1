import pandas as pd
import numpy as np
import os
import glob

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def rename_columns(df, idema):
    for col in df.columns:
        if col != 'fecha' and col not in ['anio', 'mes', 'dia', 'dia_semana', 'estacion']:
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
df_clima.to_parquet('data/processed/datos_clima/clima_numerizado_all_NaN.parquet')
df_clima.to_csv('data/processed/datos_clima/clima_numerizado_all_NaN.csv', index=False)

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
    # Ver cantidad de NaN por columna para esta estación
    nan_counts = df_clima[cols_estacion].isna().sum()
    print(f"Desglose de NaN por columna para estación {estacion}:")
    for col, count in nan_counts.items():
        print(f"  {col}: {count} NaN")

    if cols_estacion:

        horaHrMax_col = f"horaHrMax_{estacion}"
        horaHrMin_col = f"horaHrMin_{estacion}"
        
        # Impute with the mode
        if horaHrMax_col in df_clima.columns:
            mode_value = df_clima[horaHrMax_col].mode()[0]
            df_clima[horaHrMax_col] = df_clima[horaHrMax_col].fillna(mode_value)
        if horaHrMin_col in df_clima.columns:
            mode_value = df_clima[horaHrMin_col].mode()[0]
            df_clima[horaHrMin_col] = df_clima[horaHrMin_col].fillna(mode_value)

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


print("NaN en df de clima despues de procesamiento KNN")
print(df_clima.isna().sum().sum())
# # Mostrar columnas con NaN y su cantidad
# nan_cols = df_clima.columns[df_clima.isna().any()].tolist()
# print("\nColumnas con NaN:")
# for col in nan_cols:
#     nan_count = df_clima[col].isna().sum()
#     print(f"{col}: {nan_count} NaN")

# Merge con el clima
df_precio = pd.read_csv('data/processed/datos_precio/precio_consumo_electrico_timestamp_media.csv')
df_precio['timestamp'] = pd.to_datetime(df_precio['timestamp'])
df_precio['fecha'] = df_precio['timestamp'].dt.date
df_precio.drop(['consumo_kwh', 'coste_euros'], axis=1, inplace=True)

df_precio['fecha'] = pd.to_datetime(df_precio['fecha'])
df_clima['fecha'] = pd.to_datetime(df_clima['fecha'])

df_merged = pd.merge(df_precio, df_clima, on='fecha', how='left')
df_merged.dropna()
df_merged.sort_values('fecha', inplace=True)
df_merged.dropna(inplace=True)
df_merged.reset_index(drop=True, inplace=True)


print(f"Guardando el dataframe con el clima de todas las estaciones")
df_clima.to_parquet('data/processed/datos_clima/clima_numerizado_all.parquet')
df_clima.to_csv('data/processed/datos_clima/clima_numerizado_all.csv', index=False)

print(f"Guardando el dataframe con el clima y precio de todas las estaciones")
df_merged.to_parquet('data/processed/datos_precio/clima_precio_merged.parquet')
df_merged.to_csv('data/processed/datos_precio/clima_precio_merged.csv', index=False)



# Drop de columnas utiles

lista_particulas_eliminar = ['hr', 'Hr', 'indicativo', 'altitud']

for particula in lista_particulas_eliminar:
    columnas_eliminar = [col for col in df_merged.columns if particula in col]
    df_merged.drop(columnas_eliminar, axis=1, inplace=True)

df_merged.to_parquet('data/processed/datos_precio/clima_precio_merged_recortado.parquet')
df_merged.to_csv('data/processed/datos_precio/clima_precio_merged_recortado.csv', index=False)
