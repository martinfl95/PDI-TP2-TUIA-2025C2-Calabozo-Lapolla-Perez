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


def filtrar_por_agrupacion(candidatos_info):
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

    tol_h = 0.15
    tol_y = 0.7
    tol_area = 0.6

    aprobados = []

    for c in candidatos_ordenados:
        # Altura consistente
        if abs(c['h'] - mediana_h) > (mediana_h * tol_h):
            continue
        # Alineación consistente
        if abs(c['y'] - mediana_y) > (mediana_h * tol_y):
            continue
        umbral_doble = mediana_area * 1.7
        #Caracteres pegados por un pixel
        if c['area'] > umbral_doble:
            w_mitad = c['w'] // 2
            c1 = c.copy()
            c1['w'] = w_mitad
            c1['area'] = c1['w'] * c1['h']

            c2 = c.copy()
            c2['x'] = c['x'] + w_mitad
            c2['w'] = c['w'] - w_mitad
            c2['area'] = c2['w'] * c2['h']
            aprobados.append(c1)
            aprobados.append(c2)

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


def segmentar_caracteres(region_interes_color, visualizar=False):

    if region_interes_color is None or region_interes_color.size == 0:
        return [], np.zeros((10, 10), dtype=np.uint8)

    alto_roi, ancho_roi = region_interes_color.shape[:2]
    # Hacemos un recorte previo al analisis para eliminar ruido de fondo
    margen_sup = int(alto_roi * 0.10)
    margen_inf = int(alto_roi * 0.98)
    margen_izq = int(ancho_roi * 0.02)
    margen_der = int(ancho_roi * 0.98)

    # ROI
    region_interes = region_interes_color[margen_sup:margen_inf,
                                          margen_izq:margen_der]
    h_roi, w_roi = region_interes.shape[:2]
    imagen_gris = cv2.cvtColor(region_interes, cv2.COLOR_BGR2GRAY)
    # Filtro bilateral para suavizado
    imagen_filtrada = cv2.bilateralFilter(imagen_gris, 11, 17, 17)

    # Umbralado
    imagen_binaria = cv2.adaptiveThreshold(
        imagen_filtrada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9,
        0
    )
    
    # Componentes 4-conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        imagen_binaria, connectivity=4)

    candidatos_validos = []
    #Ignoramos el background, empezamos de 1 hasta num_labels
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if h == 0:
            continue
        
        #filtro de area mínima
        if area < 20:
            continue
        
        candidatos_validos.append({
            'x': x, 'y': y, 'w': w, 'h': h, 'area': area,
            'label_idx': i
        })

    #Filtrado por alineación vertical, alto de caracter y area de caracter
    candidatos_finales = filtrar_por_agrupacion(candidatos_validos)

    #Ordenados por 'x' para mantener el orden de caracteres de la patente
    candidatos_finales.sort(key=lambda c: c['x'])

    mascara_roi = np.zeros_like(imagen_gris, dtype=np.uint8)
    lista_recortes = []
    #Reconstrucción del roi patente con los bounding box de cada letra
    imagen_con_cajas = region_interes.copy()
    
    #Padding (no necesario - setear en 0 para la salida sin padding)
    pad = 1
    for c in candidatos_finales:
        x_1, y_1, w_1, h_1 = c['x'], c['y'], c['w'], c['h']
        idx = c['label_idx']
        x_pad = max(0, x_1 - pad)
        y_pad = max(0, y_1 - pad)

        x2_pad = min(w_roi, x_1 + w_1 + pad)
        y2_pad = min(h_roi, y_1 + h_1 + pad)

        #Nuevas dimensiones para visualización
        w_pad = x2_pad - x_pad
        h_pad = y2_pad - y_pad

        mascara_roi[labels == idx] = 255

        roi_recorte = region_interes[y_pad:y2_pad, x_pad:x2_pad]
        lista_recortes.append(roi_recorte)

        cv2.rectangle(imagen_con_cajas, (x_pad, y_pad),
                      (x_pad + w_pad, y_pad + h_pad), (0, 255, 0), 1)

    mascara_tamaño_completo = np.zeros((alto_roi, ancho_roi), dtype=np.uint8)

    if margen_inf > margen_sup and margen_der > margen_izq:
        mascara_tamaño_completo[margen_sup:margen_inf,
                                margen_izq:margen_der] = mascara_roi
    else:
        mascara_tamaño_completo = mascara_roi  # Fallback

    if visualizar:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(imagen_gris, cmap='gray')
        plt.title("Entrada (ROI)")
        plt.subplot(1, 4, 2)
        plt.imshow(imagen_binaria, cmap='gray')
        plt.title("Binaria")
        plt.subplot(1, 4, 3)
        plt.imshow(imagen_con_cajas)
        plt.title("Detecciones")
        plt.subplot(1, 4, 4)
        plt.imshow(mascara_roi, cmap='gray')
        plt.title(f"Mascara Final ({len(lista_recortes)})")
        plt.tight_layout()
        plt.show()

    return lista_recortes, mascara_tamaño_completo


if __name__ == '__main__':
    # Configuración de Matplotlib para todos los plots
    plt.rcParams['figure.figsize'] = [14, 8]

    lista_resultados = []

    for i in range(1, 13):
        nombre_archivo = f'img{i:02d}.png'

        recorte_patente = obtener_recorte(nombre_archivo, mostrar_pasos=False)
        if recorte_patente is not None:
            print(f"Procesando: {nombre_archivo}")

            caracteres, mascara = segmentar_caracteres(
                recorte_patente, visualizar=True)
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
