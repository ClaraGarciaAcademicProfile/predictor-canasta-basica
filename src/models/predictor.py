"""
Módulo predictor para el modelo SVM de precios de aguacate.

Este módulo contiene la clase PredictorAguacate que encapsula la lógica
de carga del modelo entrenado y las predicciones de precios futuros.
"""

import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import os
from ..utils.data_handler import crear_caracteristicas_temporales, cargar_datos_historicos


class PredictorAguacate:
    """
    Predictor de precios de aguacate usando modelo SVM entrenado.
    
    Esta clase maneja la carga del modelo SVM pre-entrenado y sus escaladores,
    y proporciona métodos para realizar predicciones de precios futuros.
    
    Atributos:
        modelo_svm: Modelo SVM cargado desde archivo.
        scaler_x: Escalador para las características de entrada.
        scaler_y: Escalador para los valores objetivo.
        columnas_caracteristicas (list): Lista de nombres de características.
        datos_historicos (pd.DataFrame): Datos históricos cargados.
        modelo_cargado (bool): Indica si el modelo fue cargado correctamente.
    """
    
    def __init__(self, ruta_modelos: str, ruta_datos: str):
        """
        Inicializa el predictor cargando el modelo y los datos históricos.
        
        Argumentos:
            ruta_modelos (str): Ruta al directorio con los archivos del modelo.
            ruta_datos (str): Ruta al archivo CSV con datos históricos.
        """
        self.ruta_modelos = ruta_modelos
        self.ruta_datos = ruta_datos
        self.modelo_svm = None
        self.scaler_x = None
        self.scaler_y = None
        self.columnas_caracteristicas = None
        self.datos_historicos = None
        self.modelo_cargado = False
        
        # Cargar automáticamente al inicializar
        self._cargar_modelo()
        self._cargar_datos_historicos()
    
    def _cargar_modelo(self) -> bool:
        """
        Carga el modelo SVM y los escaladores desde archivos.
        
        Retorna:
            bool: True si se cargó correctamente, False en caso contrario.
        """
        try:
            # Rutas a los archivos del modelo
            archivos_modelo = {
                'modelo': os.path.join(self.ruta_modelos, 'best_svm_mejorado.pkl'),
                'scaler_x': os.path.join(self.ruta_modelos, 'scaler_X.pkl'),
                'scaler_y': os.path.join(self.ruta_modelos, 'scaler_y.pkl'),
                'features': os.path.join(self.ruta_modelos, 'feature_columns.pkl')
            }
            
            # Verificar que todos los archivos existen
            for nombre, ruta in archivos_modelo.items():
                if not os.path.exists(ruta):
                    raise FileNotFoundError(f"No se encontró el archivo {nombre}: {ruta}")
            
            # Cargar los componentes del modelo
            self.modelo_svm = joblib.load(archivos_modelo['modelo'])
            self.scaler_x = joblib.load(archivos_modelo['scaler_x'])
            self.scaler_y = joblib.load(archivos_modelo['scaler_y'])
            self.columnas_caracteristicas = joblib.load(archivos_modelo['features'])
            
            self.modelo_cargado = True
            return True
            
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            self.modelo_cargado = False
            return False
    
    def _cargar_datos_historicos(self) -> bool:
        """
        Carga los datos históricos desde el archivo CSV.
        
        Retorna:
            bool: True si se cargaron correctamente, False en caso contrario.
        """
        try:
            self.datos_historicos = cargar_datos_historicos(self.ruta_datos)
            return True
        except Exception as e:
            print(f"Error al cargar datos históricos: {str(e)}")
            return False
    
    def predecir_precio_fecha(self, fecha: str) -> Dict[str, any]:
        """
        Predice el precio del aguacate para una fecha específica.
        
        Argumentos:
            fecha (str): Fecha en formato 'YYYY-MM-DD'.
            
        Retorna:
            dict: Diccionario con la predicción y metadatos.
            
        Ejemplo de uso:
            >>> resultado = predictor.predecir_precio_fecha('2024-06-15')
            >>> print(f"Precio predicho: ${resultado['precio']:.2f}")
        """
        if not self.modelo_cargado:
            return {'error': 'El modelo no está cargado correctamente'}
        
        try:
            # Validar formato de fecha
            fecha_obj = datetime.strptime(fecha, '%Y-%m-%d')
            
            # Crear DataFrame temporal con la fecha
            df_temp = pd.DataFrame({
                'Fecha_Pub_DOF': [fecha_obj],
                'Precio promedio': [np.nan]
            })
            
            # Combinar con datos históricos
            df_completo = pd.concat([
                self.datos_historicos[['Fecha_Pub_DOF', 'Precio promedio']], 
                df_temp
            ]).sort_values('Fecha_Pub_DOF').reset_index(drop=True)
            
            # Crear características
            df_features = crear_caracteristicas_temporales(df_completo)
            
            # Tomar la última fila (la fecha de predicción)
            X_pred = df_features[self.columnas_caracteristicas].iloc[-1:].fillna(
                self.datos_historicos['Precio promedio'].iloc[-1]
            )
            
            # Escalar características
            X_pred_scaled = self.scaler_x.transform(X_pred)
            
            # Realizar predicción
            pred_scaled = self.modelo_svm.predict(X_pred_scaled)
            precio_predicho = self.scaler_y.inverse_transform(
                pred_scaled.reshape(-1, 1)
            )[0][0]
            
            return {
                'fecha': fecha,
                'precio': round(precio_predicho, 2),
                'moneda': 'MXN',
                'unidad': 'kg',
                'fecha_prediccion': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'modelo_usado': 'SVM con kernel RBF'
            }
            
        except ValueError as e:
            return {'error': f'Formato de fecha inválido: {str(e)}'}
        except Exception as e:
            return {'error': f'Error en la predicción: {str(e)}'}
    
    def predecir_multiples_fechas(self, fechas: List[str]) -> List[Dict[str, any]]:
        """
        Realiza predicciones para múltiples fechas.
        
        Argumentos:
            fechas (list): Lista de fechas en formato 'YYYY-MM-DD'.
            
        Retorna:
            list: Lista de diccionarios con las predicciones.
            
        Ejemplo de uso:
            >>> fechas = ['2024-01-15', '2024-02-15', '2024-03-15']
            >>> resultados = predictor.predecir_multiples_fechas(fechas)
        """
        resultados = []
        
        for fecha in fechas:
            resultado = self.predecir_precio_fecha(fecha)
            resultados.append(resultado)
        
        return resultados
    
    def obtener_tendencia_historica(self, ultimos_meses: int = 12) -> Dict[str, any]:
        """
        Obtiene información sobre la tendencia histórica de precios.
        
        Argumentos:
            ultimos_meses (int): Número de meses hacia atrás a considerar.
            
        Retorna:
            dict: Información sobre la tendencia de precios.
        """
        if self.datos_historicos is None:
            return {'error': 'No hay datos históricos cargados'}
        
        try:
            # Filtrar datos de los últimos meses
            fecha_limite = self.datos_historicos['Fecha_Pub_DOF'].max() - pd.DateOffset(months=ultimos_meses)
            datos_recientes = self.datos_historicos[
                self.datos_historicos['Fecha_Pub_DOF'] >= fecha_limite
            ]
            
            if len(datos_recientes) == 0:
                return {'error': 'No hay suficientes datos para la tendencia'}
            
            precios = datos_recientes['Precio promedio']
            
            # Calcular tendencia simple (diferencia entre el último y primer precio)
            precio_inicial = precios.iloc[0]
            precio_final = precios.iloc[-1]
            cambio_absoluto = precio_final - precio_inicial
            cambio_porcentual = (cambio_absoluto / precio_inicial) * 100
            
            # Determinar dirección de la tendencia
            if cambio_porcentual > 5:
                direccion = 'Alcista'
            elif cambio_porcentual < -5:
                direccion = 'Bajista'
            else:
                direccion = 'Estable'
            
            return {
                'periodo_analizado': f'{ultimos_meses} meses',
                'precio_inicial': round(precio_inicial, 2),
                'precio_final': round(precio_final, 2),
                'cambio_absoluto': round(cambio_absoluto, 2),
                'cambio_porcentual': round(cambio_porcentual, 2),
                'direccion_tendencia': direccion,
                'precio_promedio_periodo': round(precios.mean(), 2),
                'precio_maximo_periodo': round(precios.max(), 2),
                'precio_minimo_periodo': round(precios.min(), 2),
                'volatilidad': round(precios.std(), 2)
            }
            
        except Exception as e:
            return {'error': f'Error al calcular tendencia: {str(e)}'}
    
    def validar_modelo(self) -> Dict[str, bool]:
        """
        Valida que todos los componentes del modelo estén cargados correctamente.
        
        Retorna:
            dict: Estado de cada componente del modelo.
        """
        return {
            'modelo_svm_cargado': self.modelo_svm is not None,
            'scaler_x_cargado': self.scaler_x is not None,
            'scaler_y_cargado': self.scaler_y is not None,
            'columnas_caracteristicas_cargadas': self.columnas_caracteristicas is not None,
            'datos_historicos_cargados': self.datos_historicos is not None,
            'modelo_completamente_funcional': self.modelo_cargado and self.datos_historicos is not None
        }
    
    def obtener_info_modelo(self) -> Dict[str, any]:
        """
        Obtiene información sobre el modelo cargado.
        
        Retorna:
            dict: Información técnica del modelo.
        """
        if not self.modelo_cargado:
            return {'error': 'Modelo no cargado'}
        
        try:
            info = {
                'tipo_modelo': 'Support Vector Regression (SVR)',
                'kernel': getattr(self.modelo_svm, 'kernel', 'No disponible'),
                'parametro_C': getattr(self.modelo_svm, 'C', 'No disponible'),
                'parametro_gamma': getattr(self.modelo_svm, 'gamma', 'No disponible'),
                'parametro_epsilon': getattr(self.modelo_svm, 'epsilon', 'No disponible'),
                'num_caracteristicas': len(self.columnas_caracteristicas) if self.columnas_caracteristicas else 0,
                'caracteristicas_usadas': self.columnas_caracteristicas if self.columnas_caracteristicas else [],
                'escalador_usado': 'MinMaxScaler',
                'datos_historicos_disponibles': len(self.datos_historicos) if self.datos_historicos is not None else 0
            }
            
            if self.datos_historicos is not None:
                info['fecha_inicio_datos'] = self.datos_historicos['Fecha_Pub_DOF'].min().strftime('%Y-%m-%d')
                info['fecha_fin_datos'] = self.datos_historicos['Fecha_Pub_DOF'].max().strftime('%Y-%m-%d')
            
            return info
            
        except Exception as e:
            return {'error': f'Error al obtener información del modelo: {str(e)}'}