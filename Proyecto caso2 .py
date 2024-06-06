import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Variables globales para almacenar los parámetros seleccionados
parametro_x = ""
parametro_y = ""
modelo_regresion = None

def cargar_archivo_csv():
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
    if ruta_archivo:
        limpieza(ruta_archivo)

def limpieza(input_csv):
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)

    # Separar la columna de fecha y hora en dos columnas diferentes
    df['Fecha'] = df['fecha_hora'].str.split(' ').str[0]
    df['Hora'] = df['fecha_hora'].str.split(' ').str[1]
    df['KW'] = df['kW'].str.replace(" ", "")

    # Convertir la columna "Fecha" a un objeto de fecha
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='mixed', dayfirst=True)

    # Agregar columna "Meses" en numero
    df['Meses'] = df['Fecha'].dt.month

    # Agregar columna "Días de la Semana" 0=Lunes, 6=Domingo
    df['Días de la Semana'] = df['Fecha'].dt.day_of_week

    # Cambiar formato de hora
    df['Hora'] = pd.to_datetime(df['Hora']).dt.hour

    # Convert 'KW' column to numeric, convierte valores no numerios en NAN antes de que tire error
    df['KW'] = pd.to_numeric(df['KW'], errors='coerce')

    # Crear las columnas para promedio de 'KW'

    df['kW_promedio(Hora)'] = df.groupby('Hora')['KW'].transform('mean')
    df['kW_promedio(Meses)'] = df.groupby('Meses')['KW'].transform('mean')
    df['kW_promedio(Semana)'] = df.groupby('Días de la Semana')['KW'].transform('mean')

    # Eliminar la columna original "fecha_hora"
    df.drop(columns=['fecha_hora'], inplace=True)
    df.drop(columns=['kW'], inplace=True)
    df.drop(columns=['kVA'], inplace=True)
    df.drop(columns=['kVAr'], inplace=True)
    df.drop(columns=['A-B(V)'], inplace=True)
    df.drop(columns=['B-C(V)'], inplace=True)
    df.drop(columns=['C-A(V)'], inplace=True)
    df.drop(columns=['L-L(V)'], inplace=True)
    df.drop(columns=['Fecha'], inplace=True)

    # Generar el nombre del archivo de salida
    parts = input_csv.split('.')
    output_csv = parts[0] + '_limpioo.' + parts[1]

    # Guardar el DataFrame modificado en el archivo de salida y no escribe columna de indices en el archivo
    df.to_csv(output_csv, index=False)

    eje_y = [col for col in df.columns if col.startswith('k')]
    eje_x = [col for col in df.columns if not col.startswith('k') and not col.startswith('K')]

    # LLamar a la funcion para mostrar ventana de opciones
    mostrar_ventana_opciones(output_csv, df.columns, eje_x, eje_y)

def mostrar_ventana_opciones(output_csv, columnas, eje_x, eje_y):
    ventana_opciones = tk.Toplevel()
    ventana_opciones.title("Opciones")

    def graficar():
        global parametro_x, parametro_y
        parametro_x = lista_columna_x.get(tk.ACTIVE)
        parametro_y = lista_columna_y.get(tk.ACTIVE)
        if parametro_x and parametro_y:
            graficar_datos(output_csv, parametro_x, parametro_y)

    def trazar_l():
        global parametro_x, parametro_y
        parametro_x = lista_columna_x.get(tk.ACTIVE)
        parametro_y = lista_columna_y.get(tk.ACTIVE)
        if parametro_x and parametro_y:
            trazar_tendencia(output_csv, parametro_x, parametro_y)
    

    def predecir_v():
        global parametro_x, parametro_y,modelo_regresion
        parametro_x = lista_columna_x.get(tk.ACTIVE)
        parametro_y = lista_columna_y.get(tk.ACTIVE)
        if parametro_x and parametro_y:
            predecir_valor(output_csv, parametro_x, parametro_y)
        
    frame_columnas = tk.Frame(ventana_opciones)
    frame_columnas.pack(pady=10)

    label_columna_x = tk.Label(frame_columnas, text="Eje X:")
    label_columna_x.grid(row=0, column=0, padx=10)

    label_columna_y = tk.Label(frame_columnas, text="Eje Y:")
    label_columna_y.grid(row=0, column=1, padx=10)

    lista_columna_x = tk.Listbox(frame_columnas, selectmode=tk.SINGLE)
    lista_columna_y = tk.Listbox(frame_columnas, selectmode=tk.SINGLE)

    for columna in eje_x:
        lista_columna_x.insert(tk.END, columna)
    for columna in eje_y:
        lista_columna_y.insert(tk.END, columna)

    lista_columna_x.grid(row=1, column=0)
    lista_columna_y.grid(row=1, column=1)

    btn_grafica = tk.Button(ventana_opciones, text="Gráfica", command=graficar)
    btn_grafica.pack(pady=10)

    btn_calcular_medidas = tk.Button(ventana_opciones, text="Calcular Medidas Básicas", command=lambda: calcular_medidas(output_csv))
    btn_calcular_medidas.pack(pady=10)

    btn_tendencia = tk.Button(ventana_opciones, text="Trazar Líneas de Tendencia", command=trazar_l)
    btn_tendencia.pack(pady=10)

    btn_prediccion = tk.Button(ventana_opciones, text="Predecir Valores Futuros", command=predecir_v)  
    btn_prediccion.pack(pady=10)

def graficar_datos(output_csv, columna_x, columna_y):
    # Leer el archivo CSV limpio
    df = pd.read_csv(output_csv)
    x = df[columna_x]
    y = df[columna_y]

    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, marker='o', color='b')
    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.title(f'Gráfico de {columna_x} vs {columna_y}')
    plt.show()


def calcular_medidas(output_csv):
    # Leer el archivo CSV limpio
    df = pd.read_csv(output_csv)

    # Calcular medidas básicas
    medidas_basicas = {
        "Promedio": np.mean(df['KW']),
        "Mediana": np.median(df['KW']),
        "Mínimo": np.min(df['KW']),
        "Máximo": np.max(df['KW']),
        "Desviación Estándar": np.std(df['KW']),
    }

    # Mostrar las medidas básicas en un cuadro de diálogo
    mensaje = "Medidas Básicas:\n"
    for medida, valor in medidas_basicas.items():
        mensaje += f"{medida}: {valor}\n"

    messagebox.showinfo("Medidas Básicas", mensaje)

def trazar_tendencia(output_csv, columna_x, columna_y):
    # Leer el archivo CSV limpio
    df = pd.read_csv(output_csv)
    x = df[columna_x].values.reshape(-1, 1)
    y = df[columna_y].values.reshape(-1, 1)
    #x = df[columna_x]
    #y = df[columna_y]

    # Generar características polinómicas de grado 2
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x)

    # Crear un modelo de regresión lineal
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    # Generar nuevos datos para la línea de tendencia
    X_new = np.linspace(x.min(), x.max(), 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new)
    y_new = lin_reg.predict(X_new_poly)

 # Crear la gráfica con la línea de tendencia polinómica
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, marker='o', color='b', label='Datos del CSV')
    plt.plot(X_new, y_new, "r-", linewidth=2, label="Línea de Tendencia")
    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.title(f'Gráfico de {columna_x} vs {columna_y} con Línea de Tendencia')
    plt.legend(loc="upper left", fontsize=14)
    plt.show()

def predecir_valor(output_csv, columna_x, columna_y):
    # Leer el archivo CSV limpio
        df = pd.read_csv(output_csv)
        x = df[parametro_x].values.reshape(-1, 1)
        y = df[parametro_y].values.reshape(-1, 1)

        # Crear características polinómicas de grado 2
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(x)

        # Crear un modelo de regresión lineal
        modelo_regresion = LinearRegression()
        modelo_regresion.fit(X_poly, y)

        # Calcular el valor Y predicho para el último valor de X en los datos
        valor_x_prediccion = x[-1][0]
        valor_x_prediccion_poly = poly_features.transform(np.array([[valor_x_prediccion]]))
        valor_y_predicho = modelo_regresion.predict(valor_x_prediccion_poly)[0][0]

        # Generar nuevos datos para la línea de tendencia polinómica
        X_new = np.linspace(x.min(), x.max(), 100).reshape(100, 1)
        X_new_poly = poly_features.transform(X_new)
        y_new = modelo_regresion.predict(X_new_poly)

        # Crear una gráfica con los datos, la línea de tendencia polinómica y el valor predicho
        plt.figure(figsize=(20, 10))
        plt.scatter(x, y, marker='o', color='b', label='Datos del CSV')
        plt.plot(X_new, y_new, "r-", linewidth=2, label='Línea de Tendencia Polinómica')
        plt.scatter(valor_x_prediccion, valor_y_predicho, color='g', s=100, label='Valor Predicho')
        plt.xlabel(parametro_x)
        plt.ylabel(parametro_y)
        plt.title(f'Gráfico de {parametro_x} vs {parametro_y} con Predicción Polinómica')
        plt.legend()
        plt.show()

# Crear la ventana principal de la aplicación
ventana = tk.Tk()
ventana.title("Cargar archivo CSV y opciones")

# Boton para cargar el archivo CSV
btn_cargar = tk.Button(ventana, text="Cargar archivo CSV", command=cargar_archivo_csv)
btn_cargar.pack(pady=20)

ventana.mainloop()
