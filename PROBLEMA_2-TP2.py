import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def validar_con_canny(roi_color):
    """
    Utiliza Canny para verificar textura vertical (letras) y descartar objetos lisos.
    """
    gris = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)

    alto, ancho = bordes.shape
    alturas = [int(alto * 0.25), int(alto * 0.5), int(alto * 0.75)]

    total_cortes = 0
    lineas_validas = 0

    for y in alturas:
        fila = bordes[y, :]
        cortes = np.count_nonzero(fila)
        if cortes > 5:
            total_cortes += cortes
            lineas_validas += 1

    if lineas_validas == 0:
        return False
    promedio = total_cortes / lineas_validas
    return 10 <= promedio <= 35


def obtener_recorte(ruta_imagen, mostrar_pasos=False):
    """
    Localiza y recorta la placa patente.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return None

    alto_img, ancho_img = imagen.shape[:2]

    # ROI
    p_arriba, p_abajo, p_costado = 0.25, 0.05, 0.20
    y_ini = int(alto_img * p_arriba)
    y_fin = int(alto_img * (1 - p_abajo))
    x_ini = int(ancho_img * p_costado)
    x_fin = int(ancho_img * (1 - p_costado))
    roi = imagen[y_ini:y_fin, x_ini:x_fin]
    if roi.size == 0:
        return None

    # Preprocesamiento
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris_eq = clahe.apply(gris)
    desenfoque = cv2.GaussianBlur(gris_eq, (5, 5), 0)

    # Bordes (Sobel)
    sobelx = cv2.Sobel(desenfoque, cv2.CV_64F, 1, 0, ksize=3)
    sobelx_abs = cv2.convertScaleAbs(sobelx)

    # Umbralado (Otsu)
    _, umbralizada = cv2.threshold(
        sobelx_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfología
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    limpia = cv2.morphologyEx(umbralizada, cv2.MORPH_OPEN, kernel_vertical)

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    morfologia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel_horizontal)

    # Análisis de contornos
    contornos, _ = cv2.findContours(
        morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    area_roi = roi.shape[0] * roi.shape[1]

    for cnt in contornos:
        area_blob = cv2.contourArea(cnt)
        if area_blob < 500:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        if float(bw) / bh < 1.5:
            continue

        rectangulo = cv2.minAreaRect(cnt)
        caja = np.int32(cv2.boxPoints(rectangulo))

        d1 = np.linalg.norm(caja[0] - caja[1])
        d2 = np.linalg.norm(caja[1] - caja[2])

        if d1 > d2:
            lado_largo, lado_corto = d1, d2
            vec = caja[0] - caja[1]
        else:
            lado_largo, lado_corto = d2, d1
            vec = caja[1] - caja[2]

        if lado_corto == 0:
            continue

        relacion_aspecto = lado_largo / lado_corto
        area_rect = lado_largo * lado_corto

        if area_rect > (area_roi * 0.2):
            continue
        if not (2.0 < relacion_aspecto < 4.5):
            continue

        angulo_grados = abs(np.degrees(np.arctan2(vec[1], vec[0]))) % 180
        angulo_horiz = min(angulo_grados, abs(180 - angulo_grados))
        if angulo_horiz > 45:
            continue

        extension = area_blob / area_rect
        if extension < 0.40:
            continue

        mascara = np.zeros_like(limpia)
        cv2.drawContours(mascara, [caja], 0, 255, -1)
        val_medio = cv2.mean(limpia, mask=mascara)[0] / 255.0

        if 0.15 < val_medio < 0.95:
            # Recorte local para Canny
            roi_candidato_bgr = roi[by:by+bh, bx:bx+bw]

            if validar_con_canny(roi_candidato_bgr):
                rect_global = ((rectangulo[0][0] + x_ini, rectangulo[0][1] + y_ini),
                               (rectangulo[1][0], rectangulo[1][1]), rectangulo[2])
                candidatos.append((rect_global, relacion_aspecto))

    # Selección
    mejor_candidato = None
    mejor_puntaje = float('inf')

    for cand in candidatos:
        cand_estructura, cand_ratio = cand
        diff_vieja = abs(cand_ratio - 2.0)
        diff_nueva = abs(cand_ratio - 3.1)
        diferencia = min(diff_vieja, diff_nueva)

        if diferencia < mejor_puntaje:
            mejor_puntaje = diferencia
            mejor_candidato = cand_estructura

    # Extracción
    roi_patente = None
    if mejor_candidato:
        caja_final = np.int32(cv2.boxPoints(mejor_candidato))

        x, y, w, h = cv2.boundingRect(caja_final)
        x, y = max(0, x), max(0, y)
        w, h = min(w, imagen.shape[1] - x), min(h, imagen.shape[0] - y)

        if w > 0 and h > 0:
            roi_patente = imagen[y:y+h, x:x+w]

    # Visualización de pasos
    if mostrar_pasos:
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 3, 1)
        plt.imshow(gris_eq, cmap='gray')
        plt.title("1. ROI + CLAHE")
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(sobelx_abs, cmap='gray')
        plt.title("2. Sobel Vertical")
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(umbralizada, cmap='gray')
        plt.title("3. Binarización (Otsu)")
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.imshow(morfologia, cmap='gray')
        plt.title("4. Morfología (Limpieza+Unión)")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        viz_deteccion = imagen.copy()
        if mejor_candidato:
            box = np.int32(cv2.boxPoints(mejor_candidato))
            cv2.drawContours(viz_deteccion, [box], 0, (0, 255, 0), 3)
        plt.imshow(cv2.cvtColor(viz_deteccion, cv2.COLOR_BGR2RGB))
        plt.title("5. Detección")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        if roi_patente is not None:
            plt.imshow(cv2.cvtColor(roi_patente, cv2.COLOR_BGR2RGB))
            plt.title("6. Recorte Final")
        else:
            plt.text(0.5, 0.5, "NO DETECTADO", ha='center')
            plt.title("6. Recorte Final")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return roi_patente


def filtrar_por_agrupacion(candidatos_info, tol_h=0.15, tol_y = 0.7, tol_area=0.6):
    if not candidatos_info:
        return []

    candidatos_ordenados = sorted(candidatos_info, key=lambda c: c['x'])

    alturas = [c['h'] for c in candidatos_ordenados]
    y_coords = [c['y'] for c in candidatos_ordenados]
    areas = [c['area'] for c in candidatos_ordenados]

    if not alturas:
        return []

    mediana_h = np.median(alturas)
    mediana_y = np.median(y_coords)
    mediana_area = np.median(areas)

    aprobados = []

    for c in candidatos_ordenados:
        # Altura consistente
        if abs(c['h'] - mediana_h) > (mediana_h * tol_h):
            continue
        # Alineación consistente
        if abs(c['y'] - mediana_y) > (mediana_h * tol_y):
            continue
        
        # Area consistente
        if abs(c['area'] - mediana_area) > (mediana_area * tol_area):
            ratio = c['w'] / c['h']
            if ratio < 0.3 and abs(c['h'] - mediana_h) < (mediana_h * 0.1):
                pass
            else:
                continue

        aprobados.append(c)

    return aprobados

def extraer_roi_interno(img_color):
    """Calcula márgenes y devuelve el ROI recortado y sus coordenadas."""
    alto, ancho = img_color.shape[:2]
    margen_sup = int(alto * 0.10)
    margen_inf = int(alto * 0.98)
    margen_izq = int(ancho * 0.02)
    margen_der = int(ancho * 0.98)
    
    roi = img_color[margen_sup:margen_inf, margen_izq:margen_der]
    return roi, (margen_sup, margen_inf, margen_izq, margen_der)

def generar_recortes_y_mascara(roi_color, labels, candidatos, pad=0):
    """
    Recorre los candidatos validados, genera los recortes, 
    la máscara binaria y la imagen con bounding boxes.
    """
    h_roi, w_roi = roi_color.shape[:2]
    roi_gris = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    
    mascara_roi = np.zeros_like(roi_gris, dtype=np.uint8)
    lista_recortes = []
    img_debug = roi_color.copy()

    for c in candidatos:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        idx = c['label_idx']
        
        mascara_roi[labels == idx] = 255
        
        #Calcular coordenadas con padding
        y1 = max(0, y - pad)
        y2 = min(h_roi, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(w_roi, x + w + pad)
        
        #Guardar recorte
        roi_recorte = roi_color[y1:y2, x1:x2]
        lista_recortes.append(roi_recorte)

        #Dibujar bounding box
        cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return lista_recortes, mascara_roi, img_debug

def graficar_resultados(img_gris, img_binaria, img_debug, mascara, titulo_main, titulo_bin):
    """Genera la figura de 4 pasos con títulos personalizados."""
    plt.figure(figsize=(12, 4))
    plt.suptitle(titulo_main, fontsize=14, fontweight='bold')

    plt.subplot(1, 4, 1)
    plt.imshow(img_gris, cmap='gray')
    plt.title("Entrada (ROI)")

    plt.subplot(1, 4, 2)
    plt.imshow(img_binaria, cmap='gray')
    plt.title(titulo_bin)

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
    plt.title("Detecciones")

    plt.subplot(1, 4, 4)
    plt.imshow(mascara, cmap='gray')
    plt.title(f"Mascara Final ({len(candidatos_finales) if 'candidatos_finales' in locals() else 'Detectados'})")

    plt.tight_layout()
    plt.show()

def crear_mascara_completa(shape_orig, mascara_roi, margenes):
    """Pega la máscara del roi dentro de una máscara del tamaño original."""
    alto, ancho = shape_orig
    m_sup, m_inf, m_izq, m_der = margenes
    
    mascara_full = np.zeros((alto, ancho), dtype=np.uint8)
    if (m_inf > m_sup) and (m_der > m_izq):
        mascara_full[m_sup:m_inf, m_izq:m_der] = mascara_roi
        
    return mascara_full

def segmentar_fallback(region_interes_color, visualizar=False):
    #Extracción de roi
    roi, margenes = extraer_roi_interno(region_interes_color)
    h_roi, w_roi = roi.shape[:2]
    imagen_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #Decidimos realizar una umbralización diferente, en este caso utilizando OTSU
    #Observamos que performaba mejor para las patentes 2, 3 y 11 pero que debíamos ajustar el umbral
    #para lograr mejores resultados
    ret, _ = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Modificación del umbral
    nuevo_umbral = ret + 30
    _, imagen_binaria = cv2.threshold(imagen_gris, nuevo_umbral, 255, cv2.THRESH_BINARY)
    
    #Componentes 8-conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, connectivity=8)

    candidatos_validos = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        #Si la altura del componente es menor a 5 pixeles no es una letra
        #Lo mismo si es un area muy pequeña
        if h < 5 or area < 10: continue 
        
        candidatos_validos.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'label_idx': i})

    #Utilizamos el mismo filtro de altura, area y alineación
    candidatos_finales = filtrar_por_agrupacion(candidatos_validos)
    candidatos_finales.sort(key=lambda c: c['x'])
    
    #Reconstrucción del roi patente con los bounding box de cada letra
    recortes, mascara_roi, img_debug = generar_recortes_y_mascara(roi, labels, candidatos_finales, pad=1)
    mascara_full = crear_mascara_completa(region_interes_color.shape[:2], mascara_roi, margenes)
    
    if visualizar:
        graficar_resultados(
            imagen_gris, imagen_binaria, img_debug, mascara_roi,
            titulo_main="FALLBACK - Otsu Modificado",
            titulo_bin="Binaria (Otsu+30)"
        )

    return recortes, mascara_full


def segmentar_caracteres(region_interes_color, visualizar=False):
    if region_interes_color is None or region_interes_color.size == 0:
        return [], np.zeros((10, 10), dtype=np.uint8)

    #Generamos el roi
    roi, margenes = extraer_roi_interno(region_interes_color)
    h_roi, w_roi = roi.shape[:2]
    imagen_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Filtro bilateral para suavizado
    imagen_filtrada = cv2.bilateralFilter(imagen_gris, 11, 17, 17)
    
    # Umbralado
    imagen_binaria = cv2.adaptiveThreshold(
        imagen_filtrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 0
    )
    
    # Componentes 4-conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_binaria, connectivity=4)

    candidatos_validos = []
    #Ignoramos el background, empezamos de 1 hasta num_labels
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if h == 0: continue
        if area < 20: continue 
        #filtro de area mínima
        candidatos_validos.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'label_idx': i})
        
    #Filtrado por alineación vertical, alto de caracter y area de caracter
    candidatos_finales = filtrar_por_agrupacion(candidatos_validos)

    #Método de fallback en caso de no encontrar los 6 caracteres efectivamente
    if len(candidatos_finales) < 6:
        recortes_fb, mascara_fb = segmentar_fallback(region_interes_color, visualizar)
        #Elegimos el mejor resultado de los dos en caso de no lograr una detección total
        if len(recortes_fb) > len(candidatos_finales):
            return recortes_fb, mascara_fb
        
    #Ordenados por 'x' para mantener el orden de caracteres de la patente
    candidatos_finales.sort(key=lambda c: c['x'])

    #Reconstrucción del roi patente con los bounding box de cada letra
    recortes, mascara_roi, img_debug = generar_recortes_y_mascara(roi, labels, candidatos_finales, pad=1)
    mascara_full = crear_mascara_completa(region_interes_color.shape[:2], mascara_roi, margenes)

    #Visualización
    if visualizar:
        graficar_resultados(
            imagen_gris, imagen_binaria, img_debug, mascara_roi,
            titulo_main="FLUJO PRINCIPAL - Adaptativo",
            titulo_bin="Binaria (Adaptativo)"
        )

    return recortes, mascara_full

## Funciones Exploratorias con Trackbars

def escalar_para_visualizacion(imagen, factor=5):
    alto, ancho = imagen.shape[:2]
    return cv2.resize(imagen, (ancho * factor, alto * factor), interpolation=cv2.INTER_NEAREST)

def explorar_ajuste_adaptativo(roi_color):
    """
    Slider para umbral adaptativo
    Controles:
    - Tamaño Bloque: Tamaño del vecindario.
    - Constante C: Valor que se resta a la media.
    """
    if roi_color is None:
        print("ROI vacía")
        return

    #Preprocesamiento
    roi, _ = extraer_roi_interno(roi_color)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    #Filtro bilateral para mantener consistencia con el método de segmentación adaptativa
    filtrada = cv2.bilateralFilter(gris, 11, 17, 17)

    nombre_ventana = "Ajuste: Adaptativo (q para salir)"
    cv2.namedWindow(nombre_ventana)
    
    #Creación de trackbars para parámetros blockSize y C
    cv2.createTrackbar("Tam. Bloque", nombre_ventana, 9, 50, lambda x: None)
    cv2.createTrackbar("Constante C", nombre_ventana, 0, 50, lambda x: None)

    print(f"--- Ajustando Adaptativo ---")
    print("Presiona 'q' o 'ESC' en la ventana para confirmar y pasar a la siguiente.")

    while True:
        #Leer trackbars
        tam_bloque = cv2.getTrackbarPos("Tam. Bloque", nombre_ventana)
        constante_c = cv2.getTrackbarPos("Constante C", nombre_ventana)

        #blockSize debe ser impar y mayor que 1
        if tam_bloque < 3: tam_bloque = 3
        if tam_bloque % 2 == 0: tam_bloque += 1

        # Aplicar umbralado
        imagen_binaria = cv2.adaptiveThreshold(
            filtrada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, tam_bloque, constante_c
        )

        # Visualización con zoom
        visualizacion = escalar_para_visualizacion(imagen_binaria)
        cv2.imshow(nombre_ventana, visualizacion)
        
        #Interrumpir ejecución con 'q' o ESC
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:
            break

    cv2.destroyWindow(nombre_ventana)
    print(f"Valores finales -> Tamaño Bloque: {tam_bloque}, C: {constante_c}")


def explorar_ajuste_otsu_desplazamiento(roi_color):
    """
    Slider - Desplazamiento: Valor a sumar/restar al umbral de Otsu calculado.
    """
    if roi_color is None:
        return

    roi, _ = extraer_roi_interno(roi_color)
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #Calcular Otsu base
    umbral_otsu, _ = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    nombre_ventana = f"Ajuste: Otsu Base {int(umbral_otsu)} (q para salir)"
    cv2.namedWindow(nombre_ventana)

    # Mapeo del trackbar:
    # Valor 50  -> Desplazamiento 0
    # Valor 100  -> Desplazamiento +50
    # Valor 0  -> Desplazamiento -50
    
    cv2.createTrackbar("Desplazamiento", nombre_ventana, 80, 100, lambda x: None)

    print(f"--- Ajustando Otsu (Base: {umbral_otsu}) ---")
    print("Presiona 'q' o 'ESC' en la ventana para confirmar y pasar a la siguiente.")
    while True:
        valor_trackbar = cv2.getTrackbarPos("Desplazamiento", nombre_ventana)
        desplazamiento = valor_trackbar - 50
        nuevo_umbral = umbral_otsu + desplazamiento

        #El umbral debe estar entre 0 y 255
        nuevo_umbral = max(0, min(255, nuevo_umbral))

        #Umbralado manual con el nuevo umbral
        _, imagen_binaria = cv2.threshold(gris, nuevo_umbral, 255, cv2.THRESH_BINARY)

        #Visualización con zoom 
        visualizacion = escalar_para_visualizacion(imagen_binaria)
        
        #Escribir el umbral actual en la imagen
        texto_info = f"Umbral: {int(nuevo_umbral)} (Otsu{'+' if desplazamiento>=0 else ''}{desplazamiento})"
        cv2.putText(visualizacion, texto_info, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (127), 2)

        cv2.imshow(nombre_ventana, visualizacion)
        
        #Interrumpir ejecución con 'q' o ESC
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:
            break

    cv2.destroyWindow(nombre_ventana)

if __name__ == '__main__':
    # Configuración de Matplotlib para todos los plots
    plt.rcParams['figure.figsize'] = [14, 8]
    
    #True: Muestra los pasos para encontrar la patente
    MOSTRAR_PASOS = True
    
    #True: Muestra la segmentación de caracteres
    MOSTRAR_SEGMENTACION = True
    
    #True: Permite el analisis exploratorio de los parámetros de umbralado (adaptativo y otsu)
    TRACKBARS = False
    lista_resultados = []

    for i in range(1,13):
        nombre_archivo = f'img{i:02d}.png'

        recorte_patente = obtener_recorte(nombre_archivo, mostrar_pasos=MOSTRAR_PASOS)
        if recorte_patente is not None:
            
            print(f"Procesando: {nombre_archivo}")
            if TRACKBARS:
                explorar_ajuste_adaptativo(recorte_patente)
                explorar_ajuste_otsu_desplazamiento(recorte_patente)

            caracteres, mascara = segmentar_caracteres(
                recorte_patente, visualizar=MOSTRAR_SEGMENTACION)
            img_debug = recorte_patente.copy()
            img_debug[mascara == 255] = [0, 255, 0]

            lista_resultados.append({
                "nombre": nombre_archivo,
                "original": recorte_patente,
                "debug": img_debug,
                "chars": caracteres
            })
        else:
            print(f"[{nombre_archivo}] No se detectó patente.")

    if lista_resultados:
        n_filas = len(lista_resultados)
        n_cols = 9

        plt.figure(figsize=(15, n_filas * 2))
        # plt.suptitle("Original vs Máscara Detectada", fontsize=8)

        for idx, item in enumerate(lista_resultados):
            base = idx * n_cols

            plt.subplot(n_filas, n_cols, base + 1)
            plt.imshow(cv2.cvtColor(item['original'], cv2.COLOR_BGR2RGB))
            plt.ylabel(item['nombre'], rotation=0,
                       labelpad=40, va='center', fontsize=9)
            plt.xticks([])
            plt.yticks([])
            if idx == 0:
                plt.title("Original")

            plt.subplot(n_filas, n_cols, base + 2)
            plt.imshow(cv2.cvtColor(item['debug'], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            if idx == 0:
                plt.title("Mascara Aplicada")

            chars = item['chars']
            for j in range(min(len(chars), 7)):
                plt.subplot(n_filas, n_cols, base + 3 + j)
                plt.imshow(cv2.cvtColor(chars[j], cv2.COLOR_BGR2RGB))
                plt.axis('off')

        plt.tight_layout()
        plt.show()
