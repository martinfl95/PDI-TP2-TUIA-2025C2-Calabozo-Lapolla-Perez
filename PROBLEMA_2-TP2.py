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
            
    if lineas_validas == 0: return False
    promedio = total_cortes / lineas_validas
    return 10 <= promedio <= 35


def obtener_recorte(ruta_imagen, mostrar_pasos=False):
    """
    Localiza y recorta la placa patente.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None: return None
    
    alto_img, ancho_img = imagen.shape[:2]

    #ROI
    p_arriba, p_abajo, p_costado = 0.25, 0.05, 0.20
    y_ini = int(alto_img * p_arriba)
    y_fin = int(alto_img * (1 - p_abajo))
    x_ini = int(ancho_img * p_costado)
    x_fin = int(ancho_img * (1 - p_costado))
    roi = imagen[y_ini:y_fin, x_ini:x_fin]
    if roi.size == 0: return None

    #Preprocesamiento
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gris_eq = clahe.apply(gris)
    desenfoque = cv2.GaussianBlur(gris_eq, (5, 5), 0)

    #Bordes (Sobel)
    sobelx = cv2.Sobel(desenfoque, cv2.CV_64F, 1, 0, ksize=3)
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    
    #Umbralado (Otsu)
    _, umbralizada = cv2.threshold(sobelx_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Morfología
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    limpia = cv2.morphologyEx(umbralizada, cv2.MORPH_OPEN, kernel_vertical)
    
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    morfologia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel_horizontal)

    #Análisis de contornos
    contornos, _ = cv2.findContours(morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    area_roi = roi.shape[0] * roi.shape[1]
    
    for cnt in contornos:
        area_blob = cv2.contourArea(cnt)
        if area_blob < 500: continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0: continue
        if float(bw) / bh < 1.5: continue 

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
            
        if lado_corto == 0: continue
        
        relacion_aspecto = lado_largo / lado_corto
        area_rect = lado_largo * lado_corto
        
        if area_rect > (area_roi * 0.2): continue 
        if not (2.0 < relacion_aspecto < 4.5): continue 
        
        angulo_grados = abs(np.degrees(np.arctan2(vec[1], vec[0]))) % 180
        angulo_horiz = min(angulo_grados, abs(180 - angulo_grados))
        if angulo_horiz > 45: continue 
        
        extension = area_blob / area_rect
        if extension < 0.40: continue

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

    #Selección
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

    #Extracción
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
        
        plt.subplot(2, 3, 1); plt.imshow(gris_eq, cmap='gray'); plt.title("1. ROI + CLAHE")
        plt.axis('off')
        plt.subplot(2, 3, 2); plt.imshow(sobelx_abs, cmap='gray'); plt.title("2. Sobel Vertical")
        plt.axis('off')
        plt.subplot(2, 3, 3); plt.imshow(umbralizada, cmap='gray'); plt.title("3. Binarización (Otsu)")
        plt.axis('off')
        plt.subplot(2, 3, 4); plt.imshow(morfologia, cmap='gray'); plt.title("4. Morfología (Limpieza+Unión)")
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

    #Ordenar por posición
    candidatos_ordenados = sorted(candidatos_info, key=lambda c: c['x'])
    
    #Medianas de la altura y ancho
    alturas = [c['h'] for c in candidatos_ordenados]
    y_coords = [c['y'] for c in candidatos_ordenados]
    
    if not alturas: return []
    
    mediana_h = np.median(alturas)
    mediana_y = np.median(y_coords)
    
    # Tolerancias de alto y ancho
    tol_h = 0.15  
    tol_y = 0.60 

    aprobados = []

    for c in candidatos_ordenados:
        # Altura consistente
        if abs(c['h'] - mediana_h) > (mediana_h * tol_h):
            continue 
            
        # Alineación vertical
        if abs(c['y'] - mediana_y) > (mediana_h * tol_y):
            continue 

        aprobados.append(c)


    return aprobados

def segmentar_caracteres(region_interes_color, visualizar=False):
    if region_interes_color is None or region_interes_color.size == 0: 
        return [], np.zeros((10,10), dtype=np.uint8)

    alto_total, ancho_total = region_interes_color.shape[:2]
    
    #Recorte de margenes
    margen_sup, margen_inf = int(alto_total * 0.10), int(alto_total * 0.98)
    margen_izq, margen_der = int(ancho_total * 0.02), int(ancho_total * 0.98)
    
    region_interes = region_interes_color[margen_sup:margen_inf, margen_izq:margen_der]
    
    #Preprocesamiento
    imagen_gris = cv2.cvtColor(region_interes, cv2.COLOR_BGR2GRAY)
    
    #Suavizado y reducción de ruido
    imagen_gris = cv2.bilateralFilter(imagen_gris, 11, 17, 17)
    
    #Binarización adaptativo
    imagen_binaria = cv2.adaptiveThreshold(imagen_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 13, 5)

    #Extracción de Contornos
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos_previos = []
    alto_roi, ancho_roi = imagen_binaria.shape

    for contorno in contornos:
        #Obtener las estadísticas del contorno
        pos_x, pos_y, ancho_cont, alto_cont = cv2.boundingRect(contorno)
        area_contorno = cv2.contourArea(contorno)
        proporcion = ancho_cont / float(alto_cont)

        #Filtros Geométricos
        if area_contorno < 30: continue
        if alto_cont < 0.20 * alto_roi: continue 
        if proporcion < 0.10 or proporcion > 2: continue

        #Guardamos la información geométrica y el contorno
        candidatos_previos.append({
            'cnt': contorno,
            'x': pos_x, 'y': pos_y, 'w': ancho_cont, 'h': alto_cont,
            'area': area_contorno
        })
    #Filtro por agrupación
    candidatos_finales_info = filtrar_por_agrupacion(candidatos_previos)
    
    mascara_roi = np.zeros_like(imagen_gris, dtype=np.uint8)
    lista_recortes = []

    for candidato in candidatos_finales_info:
        #Dibujar contorno en la máscara
        cv2.drawContours(mascara_roi, [candidato['cnt']], -1, 255, thickness=cv2.FILLED)
        
        #Extraer coordenadas y recortar de la ROI original
        x, y, w_c, h_c = candidato['x'], candidato['y'], candidato['w'], candidato['h']
        lista_recortes.append(region_interes[y:y+h_c, x:x+w_c])
        
    # Reconstruir la máscara al tamaño de la imagen de entrada
    mascara_tamaño_completo = np.zeros((alto_total, ancho_total), dtype=np.uint8)
    mascara_tamaño_completo[margen_sup:margen_inf, margen_izq:margen_der] = mascara_roi
    
    if visualizar:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1); plt.imshow(imagen_gris, cmap='gray'); plt.title("1. Imagen Gris")
        plt.subplot(1, 3, 2); plt.imshow(imagen_binaria, cmap='gray'); plt.title("2. Imagen Binaria")
        plt.subplot(1, 3, 3); plt.imshow(mascara_roi, cmap='gray'); plt.title(f"3. Máscara Filtrada ({len(lista_recortes)})")
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
            
            caracteres, mascara = segmentar_caracteres(recorte_patente, visualizar=True)
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
        plt.suptitle("DEBUG SEGMENTACIÓN: Original vs Máscara Detectada", fontsize=16)
        
        for idx, item in enumerate(lista_resultados):
            base = idx * n_cols
            
            plt.subplot(n_filas, n_cols, base + 1)
            plt.imshow(cv2.cvtColor(item['original'], cv2.COLOR_BGR2RGB))
            plt.ylabel(item['nombre'], rotation=0, labelpad=40, va='center', fontsize=9)
            plt.xticks([]); plt.yticks([])
            if idx == 0: plt.title("Original")

            plt.subplot(n_filas, n_cols, base + 2)
            plt.imshow(cv2.cvtColor(item['debug'], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            if idx == 0: plt.title("Mascara Aplicada")
            
            chars = item['chars']
            for j in range(min(len(chars), 7)):
                plt.subplot(n_filas, n_cols, base + 3 + j)
                plt.imshow(cv2.cvtColor(chars[j], cv2.COLOR_BGR2RGB))
                plt.axis('off')

        plt.tight_layout()
        plt.show()