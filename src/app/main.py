"""
Aplicación principal para la predicción de precios de aguacate.

Esta aplicación de Tkinter permite a los usuarios cargar datos históricos,
visualizar tendencias y realizar predicciones de precios usando un modelo SVM.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
from typing import Optional

# Agregar el directorio padre al path para importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..models.predictor import PredictorAguacate
    from ..utils.data_handler import obtener_estadisticas_datos, validar_archivo_datos
except ImportError:
    # En caso de que no se encuentren los módulos, mostrar error
    print("Error: No se pudieron importar los módulos necesarios")


class PredictorAguacateApp:
    """
    Aplicación principal para la predicción de precios de aguacate.
    
    Esta clase implementa una interfaz gráfica completa que permite:
    - Cargar modelos de predicción y datos históricos
    - Realizar predicciones para fechas específicas
    - Ver estadísticas y tendencias históricas
    - Generar reportes de predicción
    
    Atributos:
        ventana_principal (tk.Tk): Ventana principal de la aplicación.
        predictor (PredictorAguacate): Instancia del predictor de precios.
        notebook (ttk.Notebook): Widget de pestañas para organizar funciones.
    """
    
    def __init__(self, ventana_principal: tk.Tk):
        """
        Inicializa la aplicación con la ventana principal.
        
        Argumentos:
            ventana_principal (tk.Tk): Ventana Tk existente o recién creada.
        """
        self.ventana_principal = ventana_principal
        self.predictor: Optional[PredictorAguacate] = None
        self.notebook = None
        
        # Variables de la interfaz
        self.ruta_modelos = tk.StringVar()
        self.ruta_datos = tk.StringVar()
        self.fecha_prediccion = tk.StringVar()
        self.resultado_prediccion = tk.StringVar()
        self.estado_modelo = tk.StringVar(value="Modelo no cargado")
        
        # Widgets de texto
        self.texto_resultados = None
        self.texto_analisis = None
        self.texto_ayuda = None
        
        # Configurar ventana principal
        self._configurar_ventana_principal()
        self._crear_interfaz()
        
        # Intentar cargar modelo por defecto
        self._cargar_modelo_por_defecto()
    
    def _configurar_ventana_principal(self) -> None:
        """
        Configura las propiedades básicas de la ventana principal.
        """
        self.ventana_principal.title("Predictor de Precios de Aguacate: SVM")
        self.ventana_principal.geometry("900x700")
        self.ventana_principal.resizable(True, True)
        
        # Centrar ventana en la pantalla
        self.ventana_principal.update_idletasks()
        ancho = self.ventana_principal.winfo_width()
        alto = self.ventana_principal.winfo_height()
        pos_x = (self.ventana_principal.winfo_screenwidth() // 2) - (ancho // 2)
        pos_y = (self.ventana_principal.winfo_screenheight() // 2) - (alto // 2)
        self.ventana_principal.geometry(f'{ancho}x{alto}+{pos_x}+{pos_y}')
    
    def _crear_interfaz(self) -> None:
        """
        Crea todos los elementos de la interfaz gráfica.
        """
        # Crear barra de menú
        self._crear_menu_principal()
        
        # Crear frame principal
        frame_principal = ttk.Frame(self.ventana_principal, padding="10")
        frame_principal.grid(row=0, column=0, sticky="nsew")
        
        # Configurar expansión de grid
        self.ventana_principal.grid_rowconfigure(0, weight=1)
        self.ventana_principal.grid_columnconfigure(0, weight=1)
        frame_principal.grid_rowconfigure(2, weight=1)
        frame_principal.grid_columnconfigure(0, weight=1)
        
        # Título de la aplicación
        titulo = ttk.Label(frame_principal, text="Predictor de Precios de Aguacate", 
                          font=("Arial", 16, "bold"))
        titulo.grid(row=0, column=0, pady=(0, 10))
        
        # Barra de estado
        self._crear_barra_estado(frame_principal)
        
        # Crear notebook con pestañas
        self.notebook = ttk.Notebook(frame_principal)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        
        # Crear pestañas
        self._crear_pestana_configuracion()
        self._crear_pestana_prediccion()
        self._crear_pestana_analisis()
        self._crear_pestana_ayuda()
    
    def _crear_menu_principal(self) -> None:
        """
        Crea la barra de menú principal de la aplicación.
        """
        barra_menu = tk.Menu(self.ventana_principal)
        
        # Menú Archivo
        menu_archivo = tk.Menu(barra_menu, tearoff=0)
        menu_archivo.add_command(label="Cargar Modelo", command=self.seleccionar_directorio_modelos)
        menu_archivo.add_command(label="Cargar Datos", command=self.seleccionar_archivo_datos)
        menu_archivo.add_separator()
        menu_archivo.add_command(label="Salir", command=self.ventana_principal.quit)
        
        # Menú Herramientas
        menu_herramientas = tk.Menu(barra_menu, tearoff=0)
        menu_herramientas.add_command(label="Validar Modelo", command=self.validar_modelo)
        menu_herramientas.add_command(label="Limpiar Resultados", command=self.limpiar_resultados)
        menu_herramientas.add_command(label="Exportar Resultados", command=self.exportar_resultados)
        
        # Menú Ayuda
        menu_ayuda = tk.Menu(barra_menu, tearoff=0)
        menu_ayuda.add_command(label="Instrucciones", command=lambda: self.notebook.select(3))
        menu_ayuda.add_command(label="Acerca de", command=self.mostrar_acerca_de)
        
        # Agregar menús a la barra
        barra_menu.add_cascade(label="Archivo", menu=menu_archivo)
        barra_menu.add_cascade(label="Herramientas", menu=menu_herramientas)
        barra_menu.add_cascade(label="Ayuda", menu=menu_ayuda)
        
        self.ventana_principal.config(menu=barra_menu)
    
    def _crear_barra_estado(self, parent) -> None:
        """
        Crea la barra de estado en la parte superior.
        
        Argumentos:
            parent: Widget padre donde se colocará la barra de estado.
        """
        frame_estado = ttk.Frame(parent)
        frame_estado.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        frame_estado.grid_columnconfigure(1, weight=1)
        
        ttk.Label(frame_estado, text="Estado del modelo:", font=("Arial", 9)).grid(row=0, column=0)
        label_estado = ttk.Label(frame_estado, textvariable=self.estado_modelo, 
                                font=("Arial", 9, "italic"), foreground="blue")
        label_estado.grid(row=0, column=1, sticky="w", padx=(10, 0))
    
    def _crear_pestana_configuracion(self) -> None:
        """
        Crea la pestaña de configuración del modelo y datos.
        """
        frame_config = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_config, text="Configuración")
        
        # Sección de modelo
        grupo_modelo = ttk.LabelFrame(frame_config, text="Configuración del Modelo", padding="10")
        grupo_modelo.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        grupo_modelo.grid_columnconfigure(1, weight=1)
        
        ttk.Label(grupo_modelo, text="Directorio del modelo:").grid(row=0, column=0, sticky="w")
        entry_modelos = ttk.Entry(grupo_modelo, textvariable=self.ruta_modelos, width=50)
        entry_modelos.grid(row=0, column=1, sticky="ew", padx=(10, 10))
        ttk.Button(grupo_modelo, text="Seleccionar", 
                  command=self.seleccionar_directorio_modelos).grid(row=0, column=2)
        
        # Sección de datos
        grupo_datos = ttk.LabelFrame(frame_config, text="Datos Históricos", padding="10")
        grupo_datos.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        grupo_datos.grid_columnconfigure(1, weight=1)
        
        ttk.Label(grupo_datos, text="Archivo de datos:").grid(row=0, column=0, sticky="w")
        entry_datos = ttk.Entry(grupo_datos, textvariable=self.ruta_datos, width=50)
        entry_datos.grid(row=0, column=1, sticky="ew", padx=(10, 10))
        ttk.Button(grupo_datos, text="Seleccionar", 
                  command=self.seleccionar_archivo_datos).grid(row=0, column=2)
        
        # Botones de acción
        frame_botones = ttk.Frame(frame_config)
        frame_botones.grid(row=2, column=0, pady=(10, 0))
        
        ttk.Button(frame_botones, text="Cargar Modelo", 
                  command=self.cargar_modelo).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(frame_botones, text="Validar Archivos", 
                  command=self.validar_archivos).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(frame_botones, text="Ver Info del Modelo", 
                  command=self.mostrar_info_modelo).grid(row=0, column=2)
        
        # Área de información
        grupo_info = ttk.LabelFrame(frame_config, text="Información del Sistema", padding="10")
        grupo_info.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        frame_config.grid_rowconfigure(3, weight=1)
        
        self.texto_info = scrolledtext.ScrolledText(grupo_info, height=8, font=("Consolas", 9))
        self.texto_info.grid(row=0, column=0, sticky="nsew")
        grupo_info.grid_rowconfigure(0, weight=1)
        grupo_info.grid_columnconfigure(0, weight=1)
    
    def _crear_pestana_prediccion(self) -> None:
        """
        Crea la pestaña para realizar predicciones.
        """
        frame_pred = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_pred, text="Predicción")
        
        # Sección de entrada
        grupo_entrada = ttk.LabelFrame(frame_pred, text="Realizar Predicción", padding="10")
        grupo_entrada.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        grupo_entrada.grid_columnconfigure(1, weight=1)
        
        ttk.Label(grupo_entrada, text="Fecha (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
        entry_fecha = ttk.Entry(grupo_entrada, textvariable=self.fecha_prediccion, width=20)
        entry_fecha.grid(row=0, column=1, sticky="w", padx=(10, 10))
        
        # Botones de fecha rápida
        frame_fechas = ttk.Frame(grupo_entrada)
        frame_fechas.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        
        ttk.Button(frame_fechas, text="Hoy", 
                  command=lambda: self.fecha_prediccion.set(datetime.now().strftime('%Y-%m-%d'))).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(frame_fechas, text="+1 Mes", 
                  command=lambda: self._establecer_fecha_relativa(30)).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(frame_fechas, text="+3 Meses", 
                  command=lambda: self._establecer_fecha_relativa(90)).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(frame_fechas, text="+6 Meses", 
                  command=lambda: self._establecer_fecha_relativa(180)).grid(row=0, column=3, padx=(0, 5))
        
        # Botón de predicción
        ttk.Button(grupo_entrada, text="Predecir Precio", 
                  command=self.realizar_prediccion).grid(row=2, column=0, columnspan=2, pady=(15, 0))
        
        # Sección de resultados
        grupo_resultados = ttk.LabelFrame(frame_pred, text="Resultados de Predicción", padding="10")
        grupo_resultados.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        frame_pred.grid_rowconfigure(1, weight=1)
        
        self.texto_resultados = scrolledtext.ScrolledText(grupo_resultados, height=12, font=("Consolas", 10))
        self.texto_resultados.grid(row=0, column=0, sticky="nsew")
        grupo_resultados.grid_rowconfigure(0, weight=1)
        grupo_resultados.grid_columnconfigure(0, weight=1)
        
        # Botones de acción para resultados
        frame_acciones = ttk.Frame(frame_pred)
        frame_acciones.grid(row=2, column=0, pady=(10, 0))
        
        ttk.Button(frame_acciones, text="Limpiar Resultados", 
                  command=self.limpiar_resultados).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(frame_acciones, text="Predicciones Múltiples", 
                  command=self.ventana_predicciones_multiples).grid(row=0, column=1)
    
    def _crear_pestana_analisis(self) -> None:
        """
        Crea la pestaña para análisis de datos históricos.
        """
        frame_analisis = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_analisis, text="Análisis")
        
        # Controles de análisis
        grupo_controles = ttk.LabelFrame(frame_analisis, text="Herramientas de Análisis", padding="10")
        grupo_controles.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        frame_botones_analisis = ttk.Frame(grupo_controles)
        frame_botones_analisis.grid(row=0, column=0, sticky="w")
        
        ttk.Button(frame_botones_analisis, text="Estadísticas Básicas", 
                  command=self.mostrar_estadisticas).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(frame_botones_analisis, text="Tendencia Histórica", 
                  command=self.mostrar_tendencia).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(frame_botones_analisis, text="Análisis de Volatilidad", 
                  command=self.analizar_volatilidad).grid(row=0, column=2)
        
        # Área de resultados del análisis
        grupo_resultados_analisis = ttk.LabelFrame(frame_analisis, text="Resultados del Análisis", padding="10")
        grupo_resultados_analisis.grid(row=1, column=0, sticky="nsew")
        frame_analisis.grid_rowconfigure(1, weight=1)
        
        self.texto_analisis = scrolledtext.ScrolledText(grupo_resultados_analisis, height=20, font=("Consolas", 9))
        self.texto_analisis.grid(row=0, column=0, sticky="nsew")
        grupo_resultados_analisis.grid_rowconfigure(0, weight=1)
        grupo_resultados_analisis.grid_columnconfigure(0, weight=1)
    
    def _crear_pestana_ayuda(self) -> None:
        """
        Crea la pestaña de ayuda e instrucciones.
        """
        frame_ayuda = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame_ayuda, text="Ayuda")
        
        self.texto_ayuda = scrolledtext.ScrolledText(frame_ayuda, height=25, font=("Arial", 10))
        self.texto_ayuda.grid(row=0, column=0, sticky="nsew")
        frame_ayuda.grid_rowconfigure(0, weight=1)
        frame_ayuda.grid_columnconfigure(0, weight=1)
        
        # Contenido de ayuda
        contenido_ayuda = """
Predictior de precio de aguacate (guía)

Configuración inicial

Dirígete a la pestaña de configuración del sistema
selecciona el directorio donde tienes guardados los archivos del modelo svm
necesitarás estos archivos: best_svm_mejorado.pkl, scaler_X.pkl, scaler_y.pkl y feature_columns.pkl
también selecciona el archivo csv con los datos históricos (aguacate.csv)
presiona el botón cargar modelo para que el sistema se inicialice correctamente

Realizar predicciones

cambia a la pestaña de predicción
escribe la fecha que necesites en formato año-mes-día (por ejemplo: 2024-12-15)
si quieres ir más rápido, usa los botones de fecha que agregan 1 mes, 3 meses o el tiempo que necesites
haz clic en predecir precio y espera a que aparezca tu estimación
los resultados aparecerán en la parte de abajo de la pantalla


Análisis de datos

ve a la pestaña de análisis para explorar la información
con estadísticas básicas puedes ver un resumen de todos los datos históricos
la opción de tendencia histórica te ayuda a entender los patrones de precios del pasado
el análisis de volatilidad te muestra qué tanto suben y bajan los precios normalmente


Funciones extra que puedes usar

en el menú archivo encuentras opciones para cargar modelos y datos nuevos
el menú herramientas tiene opciones para validar el modelo, limpiar resultados y exportar información
predicciones múltiples te permite hacer varias estimaciones de una sola vez


Qué hacer si algo no funciona

si ves el mensaje "modelo no cargado", revisa que las rutas en configuración estén correctas
para errores de fecha, asegúrate de usar el formato año-mes-día con guiones
si las predicciones fallan, usa la opción "validar modelo" para revisar que todo esté bien

Cómo entender los resultados

todos los precios aparecen en pesos mexicanos por cada kilogramo
cada predicción incluye la fecha, el precio estimado y información técnica del modelo
las tendencias muestran si los precios suben o bajan y en qué porcentaje

Lo que necesitas para que funcione

python versión 3.8 o más reciente
las librerías pandas, numpy, scikit-learn y tkinter instaladas
los archivos del modelo svm ya entrenado
los datos históricos guardados en un archivo csv
        """
        
        self.texto_ayuda.insert(tk.END, contenido_ayuda)
        self.texto_ayuda.config(state=tk.DISABLED)
    
    def _cargar_modelo_por_defecto(self) -> None:
        """
        Intenta cargar el modelo desde rutas por defecto si existen.
        """
        # Rutas por defecto basadas en la estructura del proyecto
        rutas_por_defecto = {
            'modelos': 'src/models',
            'datos': 'data/aguacate.csv'
        }
        
        # Verificar si existen las rutas por defecto
        if os.path.exists(rutas_por_defecto['modelos']) and os.path.exists(rutas_por_defecto['datos']):
            self.ruta_modelos.set(rutas_por_defecto['modelos'])
            self.ruta_datos.set(rutas_por_defecto['datos'])
            self.cargar_modelo()
        else:
            self.actualizar_estado("Configure las rutas del modelo y datos")
            self._log_info("Sistema iniciado. Configure las rutas en la pestaña Configuración.")
    
    def actualizar_estado(self, mensaje: str) -> None:
        """
        Actualiza el mensaje de estado del modelo.
        
        Argumentos:
            mensaje (str): Mensaje a mostrar en la barra de estado.
        """
        self.estado_modelo.set(mensaje)
        self.ventana_principal.update_idletasks()
    
    def _log_info(self, mensaje: str) -> None:
        """
        Agrega un mensaje al área de información con timestamp.
        
        Argumentos:
            mensaje (str): Mensaje a registrar.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'texto_info') and self.texto_info:
            self.texto_info.insert(tk.END, f"[{timestamp}] {mensaje}\n")
            self.texto_info.see(tk.END)
    
    def _establecer_fecha_relativa(self, dias: int) -> None:
        """
        Establece una fecha relativa al día actual.
        
        Argumentos:
            dias (int): Número de días a agregar a la fecha actual.
        """
        fecha_futura = datetime.now() + timedelta(days=dias)
        self.fecha_prediccion.set(fecha_futura.strftime('%Y-%m-%d'))
    
    def seleccionar_directorio_modelos(self) -> None:
        """
        Abre un diálogo para seleccionar el directorio de modelos.
        """
        directorio = filedialog.askdirectory(
            title="Seleccionar directorio con archivos del modelo"
        )
        if directorio:
            self.ruta_modelos.set(directorio)
            self._log_info(f"Directorio de modelos seleccionado: {directorio}")
    
    def seleccionar_archivo_datos(self) -> None:
        """
        Abre un diálogo para seleccionar el archivo de datos históricos.
        """
        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo de datos históricos",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if archivo:
            self.ruta_datos.set(archivo)
            self._log_info(f"Archivo de datos seleccionado: {archivo}")
    
    def validar_archivos(self) -> None:
        """
        Valida que los archivos necesarios existan y tengan el formato correcto.
        """
        ruta_modelos = self.ruta_modelos.get()
        ruta_datos = self.ruta_datos.get()
        
        if not ruta_modelos or not ruta_datos:
            messagebox.showwarning("Advertencia", "Debe seleccionar tanto el directorio de modelos como el archivo de datos.")
            return
        
        # Validar archivos del modelo
        archivos_modelo = ['best_svm_mejorado.pkl', 'scaler_X.pkl', 'scaler_y.pkl', 'feature_columns.pkl']
        archivos_faltantes = []
        
        for archivo in archivos_modelo:
            ruta_completa = os.path.join(ruta_modelos, archivo)
            if not os.path.exists(ruta_completa):
                archivos_faltantes.append(archivo)
        
        # Validar archivo de datos
        valido_datos, mensaje_datos = validar_archivo_datos(ruta_datos)
        
        # Mostrar resultados
        resultado = "Validación de archivos:\n\n"
        
        if archivos_faltantes:
            resultado += f"Archivos de modelo faltantes:\n"
            for archivo in archivos_faltantes:
                resultado += f"   - {archivo}\n"
        else:
            resultado += "Todos los archivos del modelo encontrados\n"
        
        resultado += f"\nValidación de datos: {'✅' if valido_datos else '❌'}\n"
        resultado += f"   {mensaje_datos}\n"
        
        if not archivos_faltantes and valido_datos:
            resultado += "\n¡Todos los archivos son válidos! Puede proceder a cargar el modelo."
        
        self.texto_info.delete(1.0, tk.END)
        self.texto_info.insert(tk.END, resultado)
    
    def cargar_modelo(self) -> None:
        """
        Carga el modelo de predicción con las rutas configuradas.
        """
        ruta_modelos = self.ruta_modelos.get()
        ruta_datos = self.ruta_datos.get()
        
        if not ruta_modelos or not ruta_datos:
            messagebox.showerror("Error", "Debe configurar las rutas del modelo y datos antes de cargar.")
            return
        
        try:
            self.actualizar_estado("Cargando modelo...")
            self._log_info("Iniciando carga del modelo...")
            
            # Crear instancia del predictor
            self.predictor = PredictorAguacate(ruta_modelos, ruta_datos)
            
            # Verificar que se cargó correctamente
            validacion = self.predictor.validar_modelo()
            
            if validacion['modelo_completamente_funcional']:
                self.actualizar_estado("Modelo cargado y listo")
                self._log_info("Modelo cargado exitosamente")
                messagebox.showinfo("Éxito", "Modelo cargado correctamente. Ya puede realizar predicciones.")
            else:
                self.actualizar_estado("Error en la carga del modelo")
                self._log_info("Error al cargar el modelo")
                messagebox.showerror("Error", "No se pudo cargar el modelo completamente. Verifique los archivos.")
        
        except Exception as e:
            self.actualizar_estado("Error al cargar modelo")
            self._log_info(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error al cargar el modelo:\n{str(e)}")
    
    def realizar_prediccion(self) -> None:
        """
        Realiza una predicción de precio para la fecha especificada.
        """
        if not self.predictor:
            messagebox.showerror("Error", "Debe cargar el modelo antes de realizar predicciones.")
            return
        
        fecha = self.fecha_prediccion.get().strip()
        
        if not fecha:
            messagebox.showwarning("Advertencia", "Debe ingresar una fecha.")
            return
        
        try:
            self.actualizar_estado("Realizando predicción...")
            resultado = self.predictor.predecir_precio_fecha(fecha)
            
            if 'error' in resultado:
                messagebox.showerror("Error", resultado['error'])
                self.actualizar_estado("Error en predicción")
                return
            
            # Formatear y mostrar resultado
            texto_resultado = f"""
Predicción del precio del aguacate
{'='*50}

Fecha solicitada: {resultado['fecha']}
Precio predicho: ${resultado['precio']:.2f} {resultado['moneda']}
Unidad: por {resultado['unidad']}
Modelo usado: {resultado['modelo_usado']}
Fecha de predicción: {resultado['fecha_prediccion']}

Análisis contextual:
"""
            
            # Agregar análisis de tendencia si está disponible
            try:
                tendencia = self.predictor.obtener_tendencia_historica(6)
                if 'error' not in tendencia:
                    texto_resultado += f"""
   • Tendencia reciente (6 meses): {tendencia['direccion_tendencia']}
   • Precio promedio histórico: ${tendencia['precio_promedio_periodo']:.2f}
   • Rango de precios recientes: ${tendencia['precio_minimo_periodo']:.2f} - ${tendencia['precio_maximo_periodo']:.2f}
   • Volatilidad: ${tendencia['volatilidad']:.2f}
"""
            except:
                texto_resultado += "\n   • Análisis de tendencia no disponible"
            
            texto_resultado += f"\n{'='*50}\n"
            
            self.texto_resultados.insert(tk.END, texto_resultado)
            self.texto_resultados.see(tk.END)
            self.actualizar_estado("Predicción completada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar predicción:\n{str(e)}")
            self.actualizar_estado("Error en predicción")
    
    def ventana_predicciones_multiples(self) -> None:
        """
        Abre una ventana para realizar múltiples predicciones.
        """
        if not self.predictor:
            messagebox.showerror("Error", "Debe cargar el modelo antes de realizar predicciones.")
            return
        
        # Crear ventana secundaria
        ventana_multiple = tk.Toplevel(self.ventana_principal)
        ventana_multiple.title("Predicciones Múltiples")
        ventana_multiple.geometry("600x500")
        ventana_multiple.resizable(True, True)
        
        # Frame principal
        frame_principal = ttk.Frame(ventana_multiple, padding="10")
        frame_principal.grid(row=0, column=0, sticky="nsew")
        ventana_multiple.grid_rowconfigure(0, weight=1)
        ventana_multiple.grid_columnconfigure(0, weight=1)
        
        # Configuración de predicciones
        grupo_config = ttk.LabelFrame(frame_principal, text="Configuración de Predicciones", padding="10")
        grupo_config.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(grupo_config, text="Fecha inicial (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
        fecha_inicial = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        ttk.Entry(grupo_config, textvariable=fecha_inicial, width=15).grid(row=0, column=1, padx=(10, 20))
        
        ttk.Label(grupo_config, text="Número de meses:").grid(row=0, column=2, sticky="w")
        num_meses = tk.StringVar(value="6")
        ttk.Entry(grupo_config, textvariable=num_meses, width=10).grid(row=0, column=3, padx=(10, 0))
        
        # Botón de generar
        ttk.Button(grupo_config, text="Generar Predicciones", 
                  command=lambda: self._generar_predicciones_multiples(
                      fecha_inicial.get(), num_meses.get(), texto_multiples
                  )).grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        # Área de resultados
        grupo_resultados = ttk.LabelFrame(frame_principal, text="Resultados", padding="10")
        grupo_resultados.grid(row=1, column=0, sticky="nsew")
        frame_principal.grid_rowconfigure(1, weight=1)
        
        texto_multiples = scrolledtext.ScrolledText(grupo_resultados, height=20, font=("Consolas", 9))
        texto_multiples.grid(row=0, column=0, sticky="nsew")
        grupo_resultados.grid_rowconfigure(0, weight=1)
        grupo_resultados.grid_columnconfigure(0, weight=1)
    
    def _generar_predicciones_multiples(self, fecha_inicial: str, num_meses_str: str, texto_widget) -> None:
        """
        Genera múltiples predicciones para fechas consecutivas.
        
        Argumentos:
            fecha_inicial (str): Fecha de inicio en formato YYYY-MM-DD.
            num_meses_str (str): Número de meses como string.
            texto_widget: Widget de texto donde mostrar resultados.
        """
        try:
            num_meses = int(num_meses_str)
            if num_meses <= 0 or num_meses > 24:
                raise ValueError("El número de meses debe estar entre 1 y 24")
            
            # Generar fechas
            fecha_base = datetime.strptime(fecha_inicial, '%Y-%m-%d')
            fechas = []
            for i in range(num_meses):
                nueva_fecha = fecha_base + timedelta(days=30 * (i + 1))
                fechas.append(nueva_fecha.strftime('%Y-%m-%d'))
            
            # Realizar predicciones
            texto_widget.delete(1.0, tk.END)
            texto_widget.insert(tk.END, f"Predicciones múltiples precios del aguacate\n")
            texto_widget.insert(tk.END, f"{'='*60}\n\n")
            
            resultados = self.predictor.predecir_multiples_fechas(fechas)
            
            total_predicciones = 0
            suma_precios = 0
            
            for i, resultado in enumerate(resultados, 1):
                if 'error' in resultado:
                    texto_widget.insert(tk.END, f"{i:2d}. ERROR: {resultado['error']}\n")
                else:
                    precio = resultado['precio']
                    suma_precios += precio
                    total_predicciones += 1
                    texto_widget.insert(tk.END, 
                        f"{i:2d}. {resultado['fecha']} → ${precio:7.2f} MXN/kg\n")
            
            # Resumen estadístico
            if total_predicciones > 0:
                precio_promedio = suma_precios / total_predicciones
                texto_widget.insert(tk.END, f"\n{'='*60}\n")
                texto_widget.insert(tk.END, f"Resumen:\n")
                texto_widget.insert(tk.END, f"• Total de predicciones: {total_predicciones}\n")
                texto_widget.insert(tk.END, f"• Precio promedio proyectado: ${precio_promedio:.2f} MXN/kg\n")
                
                # Calcular tendencia proyectada
                if len([r for r in resultados if 'error' not in r]) >= 2:
                    precios_validos = [r['precio'] for r in resultados if 'error' not in r]
                    precio_inicial = precios_validos[0]
                    precio_final = precios_validos[-1]
                    cambio_proyectado = ((precio_final - precio_inicial) / precio_inicial) * 100
                    
                    texto_widget.insert(tk.END, f"• Cambio proyectado: {cambio_proyectado:+.1f}%\n")
                    
                    if cambio_proyectado > 5:
                        tendencia_txt = "Alcista (precios en aumento)"
                    elif cambio_proyectado < -5:
                        tendencia_txt = "Bajista (precios en descenso)"
                    else:
                        tendencia_txt = "Estable (precios relativamente constantes)"
                    
                    texto_widget.insert(tk.END, f"• Tendencia proyectada: {tendencia_txt}\n")
            
            texto_widget.see(tk.END)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Valor inválido: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar predicciones múltiples:\n{str(e)}")
    
    def mostrar_estadisticas(self) -> None:
        """
        Muestra estadísticas básicas de los datos históricos.
        """
        if not self.predictor or not self.predictor.datos_historicos is not None:
            messagebox.showerror("Error", "Debe cargar los datos históricos primero.")
            return
        
        try:
            stats = obtener_estadisticas_datos(self.predictor.datos_historicos)
            
            texto_stats = f"""
Estadíscticas básicas
{'='*50}

Info general:
• Total de registros: {stats['total_registros']}
• Período de datos: {stats['fecha_inicio']} a {stats['fecha_fin']}

Análisis de precios:
• Precio promedio: ${stats['precio_promedio']:.2f} MXN/kg
• Precio mínimo: ${stats['precio_minimo']:.2f} MXN/kg  
• Precio máximo: ${stats['precio_maximo']:.2f} MXN/kg
• Mediana: ${stats['mediana']:.2f} MXN/kg
• Desviación estándar: ${stats['desviacion_estandar']:.2f} MXN/kg

Análisis de volatilidad:
• Coeficiente de variación: {(stats['desviacion_estandar']/stats['precio_promedio'])*100:.1f}%
• Rango de precios: ${stats['precio_maximo'] - stats['precio_minimo']:.2f} MXN/kg
• Rango relativo: {((stats['precio_maximo'] - stats['precio_minimo'])/stats['precio_promedio'])*100:.1f}%

{'='*50}
"""
            
            self.texto_analisis.delete(1.0, tk.END)
            self.texto_analisis.insert(tk.END, texto_stats)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al calcular estadísticas:\n{str(e)}")
    
    def mostrar_tendencia(self) -> None:
        """
        Muestra el análisis de tendencia histórica.
        """
        if not self.predictor:
            messagebox.showerror("Error", "Debe cargar el modelo primero.")
            return
        
        try:
            # Obtener tendencias para diferentes períodos
            periodos = [3, 6, 12]
            
            texto_tendencia = f"""
Análisis de tendencia historica
{'='*50}

"""
            
            for meses in periodos:
                tendencia = self.predictor.obtener_tendencia_historica(meses)
                
                if 'error' not in tendencia:
                    texto_tendencia += f"""
Periodo: {meses} meses
{'─'*30}
• Precio inicial: ${tendencia['precio_inicial']:.2f} MXN/kg
• Precio final: ${tendencia['precio_final']:.2f} MXN/kg
• Cambio absoluto: ${tendencia['cambio_absoluto']:+.2f} MXN/kg
• Cambio porcentual: {tendencia['cambio_porcentual']:+.1f}%
• Dirección: {tendencia['direccion_tendencia']}
• Precio promedio: ${tendencia['precio_promedio_periodo']:.2f} MXN/kg
• Volatilidad: ${tendencia['volatilidad']:.2f} MXN/kg
• Rango: ${tendencia['precio_minimo_periodo']:.2f} - ${tendencia['precio_maximo_periodo']:.2f} MXN/kg

"""
                else:
                    texto_tendencia += f"\n❌ Error en período de {meses} meses: {tendencia['error']}\n"
            
            texto_tendencia += f"{'='*50}\n"
            
            self.texto_analisis.delete(1.0, tk.END)
            self.texto_analisis.insert(tk.END, texto_tendencia)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar tendencia:\n{str(e)}")
    
    def analizar_volatilidad(self) -> None:
        """
        Realiza un análisis detallado de volatilidad de precios.
        """
        if not self.predictor or self.predictor.datos_historicos is None:
            messagebox.showerror("Error", "Debe cargar los datos históricos primero.")
            return
        
        try:
            df = self.predictor.datos_historicos
            precios = df['Precio promedio']
            
            # Calcular diferentes métricas de volatilidad
            rendimientos = precios.pct_change().dropna()
            volatilidad_diaria = rendimientos.std()
            volatilidad_anualizada = volatilidad_diaria * (252 ** 0.5)  # Asumiendo 252 días de trading
            
            # Análisis de rachas
            cambios = precios.diff()
            rachas_alcistas = 0
            rachas_bajistas = 0
            racha_actual = 0
            tipo_racha = None
            
            for cambio in cambios.dropna():
                if cambio > 0:
                    if tipo_racha != 'alcista':
                        rachas_alcistas += 1
                        tipo_racha = 'alcista'
                        racha_actual = 1
                    else:
                        racha_actual += 1
                elif cambio < 0:
                    if tipo_racha != 'bajista':
                        rachas_bajistas += 1
                        tipo_racha = 'bajista'
                        racha_actual = 1
                    else:
                        racha_actual += 1
            
            # Análisis de percentiles
            p5 = precios.quantile(0.05)
            p95 = precios.quantile(0.95)
            
            texto_volatilidad = f"""
Análisis de precios
{'='*50}

Métricas de volatilidad:
• Desviación estándar: ${precios.std():.2f} MXN/kg
• Coeficiente de variación: {(precios.std()/precios.mean())*100:.1f}%
• Volatilidad de rendimientos diarios: {volatilidad_diaria*100:.2f}%
• Volatilidad anualizada: {volatilidad_anualizada*100:.1f}%

Rendimientos
• Rendimiento promedio: {rendimientos.mean()*100:+.2f}%
• Rendimiento máximo (un período): {rendimientos.max()*100:+.1f}%
• Rendimiento mínimo (un período): {rendimientos.min()*100:+.1f}%
• Días con cambios positivos: {(rendimientos > 0).sum()}
• Días con cambios negativos: {(rendimientos < 0).sum()}

Percentiles:
• Percentil 5: ${p5:.2f} MXN/kg
• Percentil 95: ${p95:.2f} MXN/kg
• Rango inter-percentil: ${p95-p5:.2f} MXN/kg

Análisis de rachas:
• Rachas alcistas identificadas: {rachas_alcistas}
• Rachas bajistas identificadas: {rachas_bajistas}

Clasificación de volatilidad:
"""
            
            # Clasificar nivel de volatilidad
            cv = (precios.std()/precios.mean())*100
            if cv < 15:
                nivel = "Baja... Mercado relativamente estable"
            elif cv < 30:
                nivel = "Moderada... Volatilidad típica de commodities"
            elif cv < 50:
                nivel = "Alta... Mercado con fluctuaciones significativas"
            else:
                nivel = "Muy alta.. Mercado extremadamente volátil"
            
            texto_volatilidad += f"* Nivel de volatilidad: {nivel}\n"
            texto_volatilidad += f"\n{'='*50}\n"
            
            self.texto_analisis.delete(1.0, tk.END)
            self.texto_analisis.insert(tk.END, texto_volatilidad)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar volatilidad:\n{str(e)}")
    
    def mostrar_info_modelo(self) -> None:
        """
        Muestra información detallada del modelo cargado.
        """
        if not self.predictor:
            messagebox.showerror("Error", "Debe cargar el modelo primero.")
            return
        
        try:
            info = self.predictor.obtener_info_modelo()
            
            if 'error' in info:
                self.texto_info.delete(1.0, tk.END)
                self.texto_info.insert(tk.END, f"Error: {info['error']}")
                return
            
            texto_info = f"""
Información del modelo SVM
{'='*50}

Configuración:
* Tipo: {info['tipo_modelo']}
* Kernel: {info['kernel']}
* Parámetro C: {info['parametro_C']}
* Parámetro Gamma: {info['parametro_gamma']}
* Parámetro Epsilon: {info['parametro_epsilon']}

Características:
* Número de características: {info['num_caracteristicas']}
• Escalador usado: {info['escalador_usado']}

Información del entrenamiento:
* Registros disponibles: {info['datos_historicos_disponibles']}
* Fecha inicio: {info.get('fecha_inicio_datos', 'N/A')}
* Fecha fin: {info.get('fecha_fin_datos', 'N/A')}

Características empleadas
"""
            
            if info['caracteristicas_usadas']:
                for i, feature in enumerate(info['caracteristicas_usadas'], 1):
                    texto_info += f"  {i:2d}. {feature}\n"
            
            texto_info += f"\n{'='*50}\n"
            
            self.texto_info.delete(1.0, tk.END)
            self.texto_info.insert(tk.END, texto_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al obtener información del modelo:\n{str(e)}")
    
    def validar_modelo(self) -> None:
        """
        Valida el estado actual del modelo.
        """
        if not self.predictor:
            messagebox.showwarning("Advertencia", "No hay modelo cargado para validar.")
            return
        
        validacion = self.predictor.validar_modelo()
        
        texto_validacion = f"""
Validación del modelo
{'='*40}

"""
        
        estados = {
            'modelo_svm_cargado': 'Modelo SVM',
            'scaler_x_cargado': 'Escalador X',
            'scaler_y_cargado': 'Escalador Y', 
            'columnas_caracteristicas_cargadas': 'Columnas de características',
            'datos_historicos_cargados': 'Datos históricos',
            'modelo_completamente_funcional': 'FUNCIONALIDAD COMPLETA'
        }
        
        for clave, descripcion in estados.items():
            estado = "✅" if validacion[clave] else "❌"
            texto_validacion += f"{estado} {descripcion}\n"
        
        if validacion['modelo_completamente_funcional']:
            texto_validacion += f"\nEl modelo está completamente funcional y listo para usar.\n"
        else:
            texto_validacion += f"\nEl modelo tiene problemas y requiere configuración adicional.\n"
        
        texto_validacion += f"\n{'='*40}\n"
        
        self.texto_info.delete(1.0, tk.END)
        self.texto_info.insert(tk.END, texto_validacion)
    
    def limpiar_resultados(self) -> None:
        """
        Limpia todas las áreas de resultados.
        """
        if self.texto_resultados:
            self.texto_resultados.delete(1.0, tk.END)
        if self.texto_analisis:
            self.texto_analisis.delete(1.0, tk.END)
        self.actualizar_estado("Resultados limpiados")
    
    def exportar_resultados(self) -> None:
        """
        Exporta los resultados actuales a un archivo de texto.
        """
        # Obtener contenido de las áreas de texto
        contenido_resultados = ""
        contenido_analisis = ""
        
        if self.texto_resultados:
            contenido_resultados = self.texto_resultados.get(1.0, tk.END).strip()
        if self.texto_analisis:
            contenido_analisis = self.texto_analisis.get(1.0, tk.END).strip()
        
        if not contenido_resultados and not contenido_analisis:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar.")
            return
        
        # Seleccionar archivo de destino
        archivo_destino = filedialog.asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if not archivo_destino:
            return
        
        try:
            with open(archivo_destino, 'w', encoding='utf-8') as f:
                f.write("Reporte de predicción de precios de aguacate\n")
                f.write(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                if contenido_resultados:
                    f.write("Resultados de predicción:\n")
                    f.write("-" * 30 + "\n")
                    f.write(contenido_resultados + "\n\n")
                
                if contenido_analisis:
                    f.write("Análisis de datos\n")
                    f.write("-" * 30 + "\n")
                    f.write(contenido_analisis + "\n\n")
                
                f.write("=" * 70 + "\n")
                f.write("Fin del reporte\n")
            
            messagebox.showinfo("Éxito", f"Resultados exportados a:\n{archivo_destino}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar resultados:\n{str(e)}")
    
    def mostrar_acerca_de(self) -> None:
        """
        Muestra información acerca de la aplicación.
        """
        mensaje = """
Predictor de Precios de Aguacate v1.0

Aplicación desarrollada como proyecto final del
Diplomado en Python para Ciencia de Datos.

Características:
• Predicción de precios usando modelo SVM
• Análisis estadístico de datos históricos
• Interfaz gráfica intuitiva con Tkinter
• Exportación de resultados
• Validación de datos y modelo
        """
        
        messagebox.showinfo("Acerca de", mensaje)


def main():
    """
    Función principal que inicia la aplicación.
    """
    # Crear ventana principal
    root = tk.Tk()
    
    # Crear aplicación
    app = PredictorAguacateApp(root)
    
    # Iniciar bucle principal
    root.mainloop()


if __name__ == "__main__":
    main()