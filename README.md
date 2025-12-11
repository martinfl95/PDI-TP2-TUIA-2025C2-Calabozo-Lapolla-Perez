# IA 4.4 Procesamiento de Imágenes - Trabajo Práctico N°2
 Facultad de Ciencias Exactas, Ingeniería y Agrimensura
 
 Tecnicatura Universitaria en Inteligencia Artificial

## Integrantes

* Calabozo, Nicolás
* Lapolla, Martín
* Perez, Sebastián

Este repositorio contiene la resolución del Trabajo Práctico N°2 de Procesamiento de Imágenes I. El trabajo se enfoca en la aplicación de técnicas de **segmentación**, **operaciones morfológicas**, **análisis de contornos** y uso de **descriptores geométricos** para la detección y clasificación automática de objetos.

## Estructura del Repositorio

* `PROBLEMA_1-TP2.py`: Script principal para la resolución del Problema 1 (Detección y clasificación de monedas y dados).
* `PROBLEMA_2-TP2.py`: Script principal para la resolución del Problema 2 (Detección de patentes y segmentación de caracteres).
* `monedas.jpg`: Imagen de entrada para el Problema 1.
* `img01.png` a `img12.png`: Conjunto de imágenes de vehículos para la detección de patentes del Problema 2.
* `README.md`: Este archivo.

## Requisitos Previos

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy (`numpy`)
* Matplotlib (`matplotlib`)

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/martinfl95/PDI-TP2-TUIA-2025C2-Calabozo-Lapolla-Perez.git
    cd PDI-TP2-TUIA-2025C2-Calabozo-Lapolla-Perez
    ```
2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv env
    # En Windows
    .\env\Scripts\activate
    # En Linux
    source env/bin/activate
    ```

## Instrucciones de Uso

Asegurarse de tener todas las imágenes necesarias (`.png`, `.jpg`) en el mismo directorio que los scripts o en la ruta esperada por el código.

### Problema 1: Detección y clasificación de monedas y dados

Este script procesa la imagen `monedas.jpg` para segmentar los objetos del fondo. Utiliza descriptores de forma ($P^2/A$) para distinguir entre monedas y dados, y luego clasifica las monedas por tamaño y cuenta el valor de los dados mediante la detección de sus puntos internos.

Para ejecutar el script:

```bash
python PROBLEMA_1-TP2.py
```

### Problema 2: Detección de patentes

Este script procesa el lote de 12 imágenes de vehículos.

Parte A: Detecta y recorta la región de la patente utilizando operaciones morfológicas y detección de bordes.

Parte B: Procesa el recorte para segmentar los caracteres individuales, utilizando técnicas de umbralado adaptativo, morfología matemática y filtrado por relación de aspecto.

Para ejecutar el script:

```bash
python PROBLEMA_2-TP2.py
