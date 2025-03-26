import pandas as pd
import numpy as np
import os
import glob

# --- Función para crear las columnas temporales	
def extract_temporal_features(df, date_column='fecha'):
    """
    Extract temporal features from a dataframe with a date column.
    
    Args:
        df (pd.DataFrame): Dataframe containing the date column
        date_column (str): Name of the date column to process
        
    Returns:
        pd.DataFrame: Dataframe with added temporal features
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Convert to datetime if not already
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract temporal features
    df['anio'] = df[date_column].dt.year
    df['mes'] = df[date_column].dt.month
    df['dia'] = df[date_column].dt.day
    df['dia_semana'] = df[date_column].dt.dayofweek
    df['estacion'] = df[date_column].dt.quarter
    
    return df

# Función para pasar de horas(String) a minutos(int)
def convertir_hora_a_minutos(x):
    if pd.isna(x) or x == 'nan':
        return pd.NA
    
    if x == 'Varias':
        # -1 podría ser un marcador para 'Varias'
        # (lo normalizaremos después, así que el -1 no afectará al modelo)
        return -1
        
    # Solo proceder si tiene el formato HH:MM esperado
    if isinstance(x, str) and ':' in x and x.replace(':', '').isdigit():
        try:
            horas, minutos = x.split(':')
            return int(horas) * 60 + int(minutos)
        except (ValueError, TypeError):
            return pd.NA
    else:
        return pd.NA
# Función para convertir minutos a seno y coseno
def minutos_a_seno_coseno(minutos_array):
    """
    Convierte minutos a valores de seno y coseno para capturar la naturaleza cíclica del tiempo.
    
    Args:
        minutos_array: Array o serie de minutos (puede contener NaN o -1)
        
    Returns:
        tuple: (array_seno, array_coseno) con valores de seno y coseno
    """
    max_minutes = 24 * 60  # Total minutos en un día
    
    # Inicializar arrays con NaN
    seno = np.full_like(minutos_array, np.nan, dtype=float)
    coseno = np.full_like(minutos_array, np.nan, dtype=float)
    
    # Filtrar valores válidos (no NaN y no -1)
    valid_mask = pd.notna(minutos_array) & (minutos_array != -1)
    valid_minutes = minutos_array[valid_mask]
    
    # Calcular valores trigonométricos solo para valores válidos
    if len(valid_minutes) > 0:
        # Convertir minutos a ángulos (0 a 2π)
        angles = valid_minutes.astype(float) * (2 * np.pi / max_minutes)
        seno[valid_mask] = np.sin(angles)
        coseno[valid_mask] = np.cos(angles)
    
    return seno, coseno
# --- Función para procesar columnas de tiempo en un dataframe
def procesar_horas_minutos(df, cols_tiempo=['horatmin', 'horatmax', 'horaracha', 'horaHrMax', 'horaHrMin']):
    """
    Procesa columnas de tiempo convirtiéndolas a minutos y representación cíclica.
    
    Args:
        df (pd.DataFrame): Dataframe con las columnas de tiempo
        cols_tiempo (list): Lista de columnas de tiempo a procesar
        
    Returns:
        pd.DataFrame: Dataframe con las columnas procesadas
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    for col in cols_tiempo:
        if col in df.columns:
            # Crear columna para indicar si es 'Varias'
            df[f'{col}_varias'] = (df[col] == 'Varias').astype(int)
            df[f'{col}_minutos'] = df[col].apply(convertir_hora_a_minutos)
            
            # Convertir minutos a seno y coseno
            seno, coseno = minutos_a_seno_coseno(df[f'{col}_minutos'])
            df[f'{col}_sin'] = seno
            df[f'{col}_cos'] = coseno

    df = df.drop(columns=cols_tiempo)
    
    return df

# Función para convertir horas enteras a minutos
def convertir_hora_entera(x):
    if pd.isna(x) or x == 'nan':
        return pd.NA
    
    if x == 'Varias':
        return -1
    
    try:
        # Convertir a entero (estas columnas solo tienen horas enteras)
        hora = int(x)
    
        # Manejar el caso especial de '24' (debería ser '00')
        if hora == 24:
            hora = 0
            
        # Convertir a minutos para mantener consistencia con otras columnas
        return hora * 60
    except (ValueError, TypeError):
        return pd.NA
# --- Función para convertir horas enteras a seno y coseno
def procesar_solo_horas(df, cols=['horaPresMax', 'horaPresMin']):
    """
    Procesa columnas de horas enteras convirtiéndolas a minutos y representación cíclica.
    
    Args:
        df (pd.DataFrame): Dataframe con las columnas de tiempo
        cols (list): Lista de columnas de tiempo a procesar
        
    Returns:
        pd.DataFrame: Dataframe con las columnas procesadas
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    for col in cols:
        if col in df.columns:
            # Crear columna para indicar si es 'Varias'
            df[f'{col}_varias'] = (df[col] == 'Varias').astype(int)
            df[f'{col}_minutos'] = df[col].apply(convertir_hora_entera)
            
            # Convertir minutos a seno y coseno
            seno, coseno = minutos_a_seno_coseno(df[f'{col}_minutos'])
            df[f'{col}_sin'] = seno
            df[f'{col}_cos'] = coseno
            
    df = df.drop(columns=cols)
    
    return df

# Función para convertir prec(String) a prec(float)
def convertir_precipitacion(x):
    if pd.isna(x):
        return 0.0  # Si no hay valor, asumimos que no ha llovido (0 mm)
    
    if isinstance(x, (int, float)):
        return float(x)
    
    if isinstance(x, str) and x == 'Ip':
        # Valor inapreciable: asignamos un valor pequeño (0.05 mm)
        return 0.05
        
    # Por si hay algún otro string inesperado
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0  # Si hay un error en la conversión, asumimos 0
# --- Función para convertir datos de precipitación
def procesar_precipitacion(df, cols=['prec']):
    """
    Procesa columnas de precipitación convirtiéndolas a valores numéricos.
    
    Args:
    df (pd.DataFrame): Dataframe con las columnas de precipitación
    cols (list): Lista de columnas de precipitación a procesar
    
    Returns:
    pd.DataFrame: Dataframe con las columnas procesadas
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    for col in cols:
        if col in df.columns:
            # Crear columna indicadora para precipitación inapreciable
            df['prec_inapreciable'] = df['prec'].apply(lambda x: 1 if isinstance(x, str) and x == 'Ip' else 0)
            
            # Aplicar la conversión
            df['prec_valor'] = df['prec'].apply(convertir_precipitacion)
            
            # Transformación logarítmica para manejar la naturaleza sesgada de las precipitaciones
            # Agregamos 1 para evitar log(0) y usar log(1+x)
            df['prec_log'] = np.log1p(df['prec_valor'])
            
            # Eliminamos la columna original
            df = df.drop(columns=['prec'])
    
    return df

# --- Función para convertir dirección del viento a seno y coseno
def direccion_viento_a_seno_coseno(df, col_dir='dir'):
    """
    Transforma la columna de dirección de la racha máxima.
    """
    # Reemplazar valores especiales por NaN
    df[col_dir] = df[col_dir].replace({99: np.nan, 88: np.nan})

    # Convertir a radianes
    df[col_dir] = np.deg2rad(df[col_dir] * 10)  # Convertir a grados y luego a radianes

    # Crear componentes seno y coseno
    df[col_dir + '_sin'] = np.sin(df[col_dir])
    df[col_dir + '_cos'] = np.cos(df[col_dir])

    # Eliminar la columna original
    df.drop(columns=[col_dir], inplace=True)

    return df

def procesar_archivo_clima(ruta_entrada, ruta_salida):
    """
    Procesa un archivo de datos climáticos y lo guarda en un formato específico.
    
    Args:
        ruta_entrada (str): Ruta del archivo CSV de entrada
        ruta_salida (str): Ruta base para guardar el archivo procesado (sin extensión)
    """
    # Cargar el archivo
    df = pd.read_csv(ruta_entrada)
    print(f"Archivo {ruta_entrada} cargado ✅")
    
    # Paso 1: sacar columnas de año, mes, dia, dia_semana y estacion
    df = extract_temporal_features(df)
    print("Columnas temporales convertidas ✅") 

    # Paso 2: Normalizar las horas a minutos
    df = procesar_horas_minutos(df)
    print("Columnas horas y minutos convertidas ✅") 

    # Paso 3: Normalizar las horas enteras a minutos
    df = procesar_solo_horas(df)
    print("Columnas horas enteras convertidas ✅")

    # Paso 4: Procesar la precipitación y hacer una transformación logarítmica
    df = procesar_precipitacion(df)
    print("Columnas de precipitación convertidas ✅")   

    # Paso 5: Tranformar la dirección del viento a seno y coseno
    df = direccion_viento_a_seno_coseno(df)
    print("Columna de dirección del viento convertida ✅")

    # Paso 6: Convertir columnas numéricas a tipo numérico
    lista_numerica = ['tmed', 'tmin', 'tmax', 'velmedia', 'racha', 'sol', 'presMax', 'presMin']
    for col in lista_numerica:
        # Replace comma with dot for decimal values before conversion
        if col in df.columns:
            if df[col].dtype == object:  # Only if column is string type
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("Columnas numéricas convertidas ✅")

    # Paso 7: Eliminar columnas que no se necesitan
    columnas_a_eliminar = ['nombre', 'provincia']
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
    print("Columnas eliminadas ✅")

    # Guardar el dataframe procesado asegurando que se mantengan los tipos de datos
    df.to_parquet(f"{ruta_salida}.parquet", index=False)
    df.to_csv(f"{ruta_salida}.csv", index=False)

    print(f"Archivo guardado en {ruta_salida}.parquet y {ruta_salida}.csv ✅")
    
    return df

# Guarda la ubicación actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = script_dir.split('data/')[0]
os.chdir(working_dir) # Poner la carpeta data como directorio actual
print(f"Directorio actual: {os.getcwd()}")

# Process all climate files ending with _clima_completo.csv

clima_files = glob.glob('data/processed/datos_clima/*_clima_completo.csv')

for file_path in clima_files:
    # Extract the filename without extension to use as output name
    base_name = file_path.rsplit('.', 1)[0]  # Remove extension
    output_path = base_name.replace('_completo', '_numerizado')
    
    print(f"Processing {file_path}...")
    df_procesado = procesar_archivo_clima(file_path, output_path)
    print(f"Finished processing {file_path}\n")
