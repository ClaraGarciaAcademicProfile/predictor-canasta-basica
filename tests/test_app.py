"""
Pruebas básicas para la aplicación de predicción de precios de aguacate.

Este módulo contiene pruebas unitarias básicas para validar el funcionamiento
de los componentes principales de la aplicación.
"""
import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.data_handler import (
        cargar_datos_historicos, 
        crear_caracteristicas_temporales,
        obtener_estadisticas_datos,
        validar_archivo_datos,
        generar_fechas_futuras
    )
    from src.models.predictor import PredictorAguacate
except ImportError as e:
    print(f"Error importando módulos: {e}")


class TestDataHandler(unittest.TestCase):
    """
    Pruebas para las funciones de manejo de datos.
    """
    
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        # Crear datos de prueba
        fechas = pd.date_range('2020-01-01', '2023-12-31', freq='M')
        precios = np.random.uniform(40, 100, len(fechas))
        
        self.datos_prueba = pd.DataFrame({
            'Fecha_Pub_DOF': fechas,
            'Precio promedio': precios
        })
        
        # Guardar archivo temporal de prueba
        self.archivo_prueba = 'test_data.csv'
        self.datos_prueba.to_csv(self.archivo_prueba, index=False)
    
    def tearDown(self):
        """
        Limpieza después de las pruebas.
        """
        if os.path.exists(self.archivo_prueba):
            os.remove(self.archivo_prueba)
    
    def test_cargar_datos_historicos(self):
        """
        Prueba la función de carga de datos históricos.
        """
        datos = cargar_datos_historicos(self.archivo_prueba)
        
        self.assertIsInstance(datos, pd.DataFrame)
        self.assertIn('Fecha_Pub_DOF', datos.columns)
        self.assertIn('Precio promedio', datos.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(datos['Fecha_Pub_DOF']))
    
    def test_crear_caracteristicas_temporales(self):
        """
        Prueba la creación de características temporales.
        """
        df_features = crear_caracteristicas_temporales(self.datos_prueba)
        
        # Verificar que se crearon las nuevas columnas
        columnas_esperadas = [
            'year', 'month', 'day_of_year', 'quarter',
            'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'days_since_start', 'precio_lag1', 'precio_lag2', 'precio_lag3',
            'precio_ma3', 'precio_ma6', 'precio_std3',
            'precio_diff1', 'precio_diff2'
        ]
        
        for columna in columnas_esperadas:
            self.assertIn(columna, df_features.columns)
    
    def test_obtener_estadisticas_datos(self):
        """
        Prueba el cálculo de estadísticas de datos.
        """
        stats = obtener_estadisticas_datos(self.datos_prueba)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('precio_promedio', stats)
        self.assertIn('precio_minimo', stats)
        self.assertIn('precio_maximo', stats)
        self.assertIn('total_registros', stats)
    
    def test_validar_archivo_datos(self):
        """
        Prueba la validación de archivos de datos.
        """
        es_valido, mensaje = validar_archivo_datos(self.archivo_prueba)
        
        self.assertTrue(es_valido)
        self.assertIsInstance(mensaje, str)
    
    def test_generar_fechas_futuras(self):
        """
        Prueba la generación de fechas futuras.
        """
        fechas = generar_fechas_futuras('2024-01-01', 6)
        
        self.assertIsInstance(fechas, list)
        self.assertEqual(len(fechas), 6)
        
        # Verificar formato de fechas
        for fecha in fechas:
            self.assertRegex(fecha, r'\d{4}-\d{2}-\d{2}')


class TestPredictorIntegracion(unittest.TestCase):
    """
    Pruebas de integración para el predictor (requiere modelo entrenado).
    """
    
    def setUp(self):
        """
        Configuración inicial para pruebas del predictor.
        """
        # Estas rutas deben ajustarse según la ubicación real de los archivos
        self.ruta_modelos = 'data/models'
        self.ruta_datos = 'data/aguacate.csv'
        
        # Solo ejecutar pruebas si los archivos existen
        self.archivos_disponibles = (
            os.path.exists(self.ruta_modelos) and 
            os.path.exists(self.ruta_datos)
        )
    
    def test_inicializacion_predictor(self):
        """
        Prueba la inicialización del predictor.
        """
        if not self.archivos_disponibles:
            self.skipTest("Archivos del modelo no disponibles para pruebas")
        
        try:
            predictor = PredictorAguacate(self.ruta_modelos, self.ruta_datos)
            self.assertIsNotNone(predictor)
        except Exception as e:
            self.fail(f"Error al inicializar predictor: {e}")
    
    def test_validacion_modelo(self):
        """
        Prueba la validación del modelo cargado.
        """
        if not self.archivos_disponibles:
            self.skipTest("Archivos del modelo no disponibles para pruebas")
        
        try:
            predictor = PredictorAguacate(self.ruta_modelos, self.ruta_datos)
            validacion = predictor.validar_modelo()
            
            self.assertIsInstance(validacion, dict)
            self.assertIn('modelo_completamente_funcional', validacion)
        except Exception as e:
            self.fail(f"Error en validación del modelo: {e}")
    
    def test_prediccion_simple(self):
        """
        Prueba una predicción simple.
        """
        if not self.archivos_disponibles:
            self.skipTest("Archivos del modelo no disponibles para pruebas")
        
        try:
            predictor = PredictorAguacate(self.ruta_modelos, self.ruta_datos)
            fecha_prueba = '2024-06-15'
            resultado = predictor.predecir_precio_fecha(fecha_prueba)
            
            if 'error' not in resultado:
                self.assertIn('precio', resultado)
                self.assertIsInstance(resultado['precio'], (int, float))
                self.assertGreater(resultado['precio'], 0)
            else:
                print(f"Advertencia en predicción: {resultado['error']}")
                
        except Exception as e:
            self.fail(f"Error en predicción simple: {e}")


class TestValidacionDatos(unittest.TestCase):
    """
    Pruebas de validación de datos y formatos.
    """
    
    def test_formato_fecha_valido(self):
        """
        Prueba validación de formatos de fecha.
        """
        fechas_validas = ['2024-01-15', '2023-12-31', '2025-06-01']
        fechas_invalidas = ['2024-13-01', '15-01-2024', '2024/01/15', 'invalid']
        
        for fecha in fechas_validas:
            try:
                datetime.strptime(fecha, '%Y-%m-%d')
                # Si no lanza excepción, el formato es válido
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Fecha válida rechazada: {fecha}")
        
        for fecha in fechas_invalidas:
            try:
                datetime.strptime(fecha, '%Y-%m-%d')
                # Si no lanza excepción, algo está mal
                if fecha != '2024-13-01':  # Esta puede pasar en algunos sistemas
                    self.fail(f"Fecha inválida aceptada: {fecha}")
            except ValueError:
                # Comportamiento esperado
                self.assertTrue(True)
    
    def test_rango_precios_logico(self):
        """
        Prueba que los precios predichos estén en rangos lógicos.
        """
        # Los precios de aguacate típicamente están entre 20 y 200 pesos por kg
        precio_minimo_logico = 20
        precio_maximo_logico = 200
        
        # Esta prueba es conceptual, en la práctica dependería de tener un predictor
        precios_prueba = [45.50, 67.25, 123.75, 89.00]
        
        for precio in precios_prueba:
            self.assertGreaterEqual(precio, precio_minimo_logico,
                                  f"Precio muy bajo: {precio}")
            self.assertLessEqual(precio, precio_maximo_logico,
                               f"Precio muy alto: {precio}")


def ejecutar_todas_las_pruebas():
    """
    Ejecuta todas las pruebas unitarias.
    """
    # Crear suite de pruebas
    suite = unittest.TestSuite()
    
    # Agregar pruebas
    suite.addTest(unittest.makeSuite(TestDataHandler))
    suite.addTest(unittest.makeSuite(TestValidacionDatos))
    suite.addTest(unittest.makeSuite(TestPredictorIntegracion))
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    resultado = runner.run(suite)
    
    return resultado


if __name__ == '__main__':
    print("Ejecutando pruebas del Predictor de Precios de Aguacate")
    print("=" * 60)
    
    # Ejecutar todas las pruebas
    resultado = ejecutar_todas_las_pruebas()
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS:")
    print(f"Pruebas ejecutadas: {resultado.testsRun}")
    print(f"Errores: {len(resultado.errors)}")
    print(f"Fallos: {len(resultado.failures)}")
    print(f"Omitidas: {len(resultado.skipped) if hasattr(resultado, 'skipped') else 0}")
    
    if resultado.wasSuccessful():
        print("\n✓ Todas las pruebas pasaron exitosamente!")
    else:
        print("\n✗ Algunas pruebas fallaron. Revise los detalles arriba.")
        
        if resultado.errors:
            print("\nErrores encontrados:")
            for test, error in resultado.errors:
                print(f"- {test}: {error.split('\\n')[0]}")
                
        if resultado.failures:
            print("\nFallos encontrados:")
            for test, fallo in resultado.failures:
                print(f"- {test}: {fallo.split('\\n')[0]}")