"""
Módulo para el manejo y procesamiento de datos del predictor de precios de aguacate.

Este módulo contiene las funciones necesarias para cargar, procesar y preparar
los datos para el modelo de predicción de precios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import os


def cargar_datos_historicos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga los datos históricos de precios de aguacate desde un archivo CSV.
    
    Argumentos:
        ruta_archivo (str): Ruta al archivo CSV con los datos históricos.
        
    Retorna:
        pd.DataFrame: DataFrame con las columnas 'Fecha_Pub_DOF' y 'Precio promedio'.
        
    Ejemplo de uso:
        >>> datos = cargar_datos_historicos('data/aguacate.csv')
        >>> print(datos.head())
    """
    try:
        # Cargar el CSV con las columnas de fecha y precio
        df = pd.read_csv(ruta_archivo)
        
        # Convertir la columna de fecha al formato datetime
        df['Fecha_Pub_DOF'] = pd.to_datetime(df['Fecha_Pub_DOF'])
        
        # Ordenar por fecha
        df = df.sort_values('Fecha_Pub_DOF').reset_index(drop=True)
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
    except Exception as e:
        raise Exception(f"Error al cargar los datos: {str(e)}")


def crear_caracteristicas_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características temporales y de lag a partir de los datos históricos.
    
    Argumentos:
        df (pd.DataFrame): DataFrame con columnas 'Fecha_Pub_DOF' y 'Precio promedio'.
        
    Retorna:
        pd.DataFrame: DataFrame con características adicionales para el modelo.
        
    Ejemplo de uso:
        >>> df_features = crear_caracteristicas_temporales(datos)
        >>> print(df_features.columns.tolist())
    """
    df_features = df.copy()
    
    # Características temporales básicas
    df_features['year'] = df_features['Fecha_Pub_DOF'].dt.year
    df_features['month'] = df_features['Fecha_Pub_DOF'].dt.month
    df_features['day_of_year'] = df_features['Fecha_Pub_DOF'].dt.dayofyear
    df_features['quarter'] = df_features['Fecha_Pub_DOF'].dt.quarter
    
    # Características cíclicas para capturar estacionalidad
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
    
    # Características de tendencia temporal
    df_features['days_since_start'] = (df_features['Fecha_Pub_DOF'] - 
                                     df_features['Fecha_Pub_DOF'].min()).dt.days
    
    # Características de lag (precios anteriores)
    df_features['precio_lag1'] = df_features['Precio promedio'].shift(1)
    df_features['precio_lag2'] = df_features['Precio promedio'].shift(2)
    df_features['precio_lag3'] = df_features['Precio promedio'].shift(3)
    
    # Medias móviles
    df_features['precio_ma3'] = df_features['Precio promedio'].rolling(window=3).mean()
    df_features['precio_ma6'] = df_features['Precio promedio'].rolling(window=6).mean()
    
    # Volatilidad (desviación estándar móvil)
    df_features['precio_std3'] = df_features['Precio promedio'].rolling(window=3).std()
    
    # Diferencias (cambios en precio)
    df_features['precio_diff1'] = df_features['Precio promedio'].diff()
    df_features['precio_diff2'] = df_features['Precio promedio'].diff(2)
    
    return df_features


def obtener_estadisticas_datos(df: pd.DataFrame) -> dict:
    """
    Calcula estadísticas básicas de los datos históricos de precios.
    
    Argumentos:
        df (pd.DataFrame): DataFrame con los datos de precios.
        
    Retorna:
        dict: Diccionario con estadísticas básicas.
        
    Ejemplo de uso:
        >>> stats = obtener_estadisticas_datos(datos)
        >>> print(f"Precio promedio: ${stats['precio_promedio']:.2f}")
    """
    if 'Precio promedio' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Precio promedio'")
    
    precios = df['Precio promedio']
    
    estadisticas = {
        'total_registros': len(df),
        'fecha_inicio': df['Fecha_Pub_DOF'].min().strftime('%Y-%m-%d'),
        'fecha_fin': df['Fecha_Pub_DOF'].max().strftime('%Y-%m-%d'),
        'precio_promedio': precios.mean(),
        'precio_minimo': precios.min(),
        'precio_maximo': precios.max(),
        'desviacion_estandar': precios.std(),
        'mediana': precios.median()
    }
    
    return estadisticas


def validar_archivo_datos(ruta_archivo: str) -> Tuple[bool, str]:
    """
    Valida que el archivo de datos existe y tiene el formato correcto.
    
    Argumentos:
        ruta_archivo (str): Ruta al archivo a validar.
        
    Retorna:
        tuple: (es_valido, mensaje)
        
    Ejemplo de uso:
        >>> valido, mensaje = validar_archivo_datos('data/aguacate.csv')
        >>> if not valido:
        >>>     print(mensaje)
    """
    # Verificar que el archivo existe
    if not os.path.exists(ruta_archivo):
        return False, f"El archivo no existe: {ruta_archivo}"
    
    try:
        # Intentar cargar el archivo
        df = pd.read_csv(ruta_archivo)
        
        # Verificar que tiene las columnas necesarias
        columnas_requeridas = ['Fecha_Pub_DOF', 'Precio promedio']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            return False, f"Faltan las columnas: {', '.join(columnas_faltantes)}"
        
        # Verificar que hay datos
        if len(df) == 0:
            return False, "El archivo está vacío"
        
        # Verificar que los precios son válidos
        if df['Precio promedio'].isnull().all():
            return False, "No hay datos de precios válidos"
        
        return True, "Archivo válido"
        
    except Exception as e:
        return False, f"Error al leer el archivo: {str(e)}"


def generar_fechas_futuras(fecha_inicio: str, num_meses: int) -> List[str]:
    """
    Genera una lista de fechas futuras para predicciones.
    
    Argumentos:
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        num_meses (int): Número de meses hacia el futuro.
        
    Retorna:
        list: Lista de fechas en formato string.
        
    Ejemplo de uso:
        >>> fechas = generar_fechas_futuras('2024-01-15', 6)
        >>> print(fechas)
    """
    try:
        fecha_base = datetime.strptime(fecha_inicio, '%Y-%m-%d')
        fechas_futuras = []
        
        for i in range(num_meses):
            # Agregar meses (aproximadamente 30 días cada uno)
            nueva_fecha = fecha_base + timedelta(days=30 * (i + 1))
            fechas_futuras.append(nueva_fecha.strftime('%Y-%m-%d'))
        
        return fechas_futuras
        
    except ValueError as e:
        raise ValueError(f"Formato de fecha inválido. Use 'YYYY-MM-DD': {str(e)}")