import cv2
import numpy as np
import matplotlib.pyplot as plt

def validar_con_canny(roi_color):
    """
    Utiliza el detector de bordes Canny para verificar si el recorte candidato
    posee la textura interna (bordes verticales de letras/números) típica de una patente.
    Ayuda a descartar falsos positivos lisos como ópticas o espejos.
    """
    # 1. Convertir recorte a escala de grises
    gris = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar Canny para detectar bordes fuertes
    # Los umbrales 50-150 son estándar para detectar trazos definidos
    bordes = cv2.Canny(gris, 50, 150)
    
    # 3. Escanear 3 líneas horizontales a distintas alturas (25%, 50%, 75%)
    # Esto se hace para asegurar que cruzamos las letras en algún punto
    alto, ancho = bordes.shape
    alturas = [int(alto * 0.25), int(alto * 0.5), int(alto * 0.75)]
    
    total_cortes = 0
    lineas_validas = 0
    
    for y in alturas:
        fila = bordes[y, :]
        # Contamos píxeles blancos (bordes verticales interceptados)
        cortes = np.count_nonzero(fila)
        
        # Una línea de patente válida debe cruzar varias letras.
        # 6 caracteres * 2 bordes = 12 bordes mínimo aprox. Damos margen > 5.
        if cortes > 5: 
            total_cortes += cortes
            lineas_validas += 1
            
    # Si ninguna línea encontró bordes, no es patente
    if lineas_validas == 0: return False
    
    promedio = total_cortes / lineas_validas

    # RANGO DE ACEPTACIÓN:
    # - Menos de 10: Probablemente un objeto liso o logo simple.
    # - Más de 30: Probablemente ruido, pasto o parrilla muy densa.
    # - Entre 10 y 30: Rango típico de texto de patente.
    return 10 <= promedio <= 30

def obtener_recorte_sin_padding(ruta_imagen, visualizar=False):
    """
    Localiza y recorta la placa patente utilizando morfología matemática y análisis de contornos.
    """
    # --- CARGA Y RECORTE INICIAL (ROI) ---
    imagen = cv2.imread(ruta_imagen)
    if imagen is None: return None
    alto_img, ancho_img = imagen.shape[:2]

    # Definimos porcentajes para ignorar bordes irrelevantes (cielo, piso, costados extremos)
    p_arriba, p_abajo, p_costado = 0.25, 0.05, 0.20
    y_ini = int(alto_img * p_arriba)
    y_fin = int(alto_img * (1 - p_abajo))
    x_ini = int(ancho_img * p_costado)
    x_fin = int(ancho_img * (1 - p_costado))
    
    # Extraemos la Región de Interés (ROI) general
    roi = imagen[y_ini:y_fin, x_ini:x_fin]
    if roi.size == 0: return None

    # --- PREPROCESAMIENTO ---
    # 1. Escala de grises
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Mejora de contraste local (CLAHE) para resaltar letras en sombras
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gris = clahe.apply(gris)
    
    # 3. Suavizado Gaussiano para reducir ruido antes de detectar bordes
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)

    # 4. Detección de bordes verticales (Sobel X)
    # Las patentes tienen alto contraste vertical (letras vs fondo)
    sobelx = cv2.Sobel(desenfoque, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    
    # 5. Binarización (Otsu) para separar bordes del fondo
    _, umbralizada = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- MORFOLOGÍA ---
    # 1. Limpieza vertical: Eliminar ruido pequeño (líneas finas horizontales)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    limpia = cv2.morphologyEx(umbralizada, cv2.MORPH_OPEN, kernel_vertical)
    
    # 2. Fusión horizontal: Conectar las letras para formar un solo bloque rectangular
    # El kernel (13, 3) es clave: ancho para unir letras, bajo para no unir con parrillas
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    morfologia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel_horizontal)

    # --- ANÁLISIS DE CONTORNOS ---
    contornos, _ = cv2.findContours(morfologia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    area_roi = roi.shape[0] * roi.shape[1]

    # Copia para visualización (solo si se pide)
    if visualizar:
        viz_img = imagen.copy()
        cv2.rectangle(viz_img, (x_ini, y_ini), (x_fin, y_fin), (255, 0, 0), 2)

    for cnt in contornos:
        # 1. Filtro de Área Mínima: Descartar ruido pequeño
        area_blob = cv2.contourArea(cnt)
        if area_blob < 500:
            continue

        # 2. Filtro de Proporción del Bounding Box recto
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0: continue
        #Si es más alto que ancho (vertical), no es patente
        if float(bw) / bh < 1.5:
            continue 

        #Obtener el rectángulo rotado (ajustado a la orientación del objeto)
        rectangulo = cv2.minAreaRect(cnt)
        caja = np.int32(cv2.boxPoints(rectangulo))
        
        #Calcular dimensiones reales (largo y corto) independientemente de la rotación
        d1 = np.linalg.norm(caja[0] - caja[1])
        d2 = np.linalg.norm(caja[1] - caja[2])
        
        if d1 > d2:
            lado_largo, lado_corto = d1, d2
            vec = caja[0] - caja[1]
        else:
            lado_largo, lado_corto = d2, d1
            vec = caja[1] - caja[2]
            
        if lado_corto == 0: continue
        
        #Filtros geometricos
        
        #Relación de Aspecto (Largo / Corto)
        relacion_aspecto = lado_largo / lado_corto
        
        #Área máxima relativa (no puede ocupar más del 20% de la imagen)
        area_rect = lado_largo * lado_corto
        if area_rect > (area_roi * 0.2): continue 
        
        #Rango de Aspect Ratio permitido (Patente vieja ~2.0, Nueva ~3.1)
        if not (2.0 < relacion_aspecto < 4): continue 
        
        #Ángulo de inclinación (no aceptamos patentes muy torcidas > 45°)
        angulo_grados = abs(np.degrees(np.arctan2(vec[1], vec[0]))) % 180
        angulo_horiz = min(angulo_grados, abs(180 - angulo_grados))
        if angulo_horiz > 45: continue 
        
        # Rectangularidad (Extent): Área real / Área del rectángulo envolvente
        # Un valor bajo (< 0.49) indica que el objeto está muy "hueco" o irregular
        extension = area_blob / area_rect
        if extension < 0.40: continue

        # --- VALIDACIONES FINALES ---
        
        # Intensidad media dentro de la máscara (evitar bloques negros o blancos puros)
        mascara = np.zeros_like(limpia)
        cv2.drawContours(mascara, [caja], 0, 255, -1)
        val_medio = cv2.mean(limpia, mask=mascara)[0] / 255.0
        
        if 0.15 < val_medio < 0.95:
            # Validación de Textura con Canny
            roi_candidato_bgr = cv2.cvtColor(gris[by:by+bh, bx:bx+bw], cv2.COLOR_GRAY2BGR)

            if validar_con_canny(roi_candidato_bgr):
                # Si pasa todo, guardamos el candidato ajustando coordenadas globales
                rect_global = ((rectangulo[0][0] + x_ini, rectangulo[0][1] + y_ini), 
                               (rectangulo[1][0], rectangulo[1][1]), rectangulo[2])
                candidatos.append((rect_global, relacion_aspecto))

    # Selección de candidatos
    # Buscamos el candidato cuyo ratio se acerque más a los estándares (2.0 o 3.1)
    mejor_candidato = None
    mejor_puntaje = 1000 # Mientras menor sea, mejor
    
    for cand in candidatos:
        cand_estructura, cand_ratio = cand
        
        diff_vieja = abs(cand_ratio - 2.0)  # Patente Vieja
        diff_nueva = abs(cand_ratio - 3.1)  # Patente Mercosur

        diferencia = min(diff_vieja, diff_nueva)
        if diferencia < mejor_puntaje:
            mejor_puntaje = diferencia
            mejor_candidato = cand_estructura

    # Extracción
    roi_patente = None
    if mejor_candidato:
        caja_final = np.int32(cv2.boxPoints(mejor_candidato))
        
        if visualizar:
            cv2.drawContours(viz_img, [caja_final], 0, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(caja_final)
        # Aseguramos límites dentro de la imagen
        x, y = max(0, x), max(0, y)
        w, h = min(w, imagen.shape[1] - x), min(h, imagen.shape[0] - y)
        
        if w > 0 and h > 0:
            roi_patente = imagen[y:y+h, x:x+w]

    # Visualización paso a paso (solo debug)
    if visualizar:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.title("Detección en Imagen"); plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2); plt.title("Recorte Final"); 
        if roi_patente is not None:
            plt.imshow(cv2.cvtColor(roi_patente, cv2.COLOR_BGR2RGB))
        else:
            plt.text(0.5, 0.5, "No detectado", ha='center')
        plt.show()

    return roi_patente

#if __name__ == '__main__':
#    lista_patentes = []
#
#    for i in range(1, 13):
#        nombre_archivo = f'img{i:02d}.png'
#        recorte = obtener_recorte_sin_padding(nombre_archivo, visualizar=True) 
#        
#        if recorte is not None:
#            lista_patentes.append({"nombre": nombre_archivo, "imagen": recorte})
#            print(f"[{nombre_archivo}] Detectada.")
#        else:
#            print(f"[{nombre_archivo}] No detectada.")
#
#    # Visualización
#    if lista_patentes:
#        cols = 4
#        filas = (len(lista_patentes) // cols) + 1
#        plt.figure(figsize=(15, filas * 3))
#        
#        for idx, item in enumerate(lista_patentes):
#            plt.subplot(filas, cols, idx + 1)
#            plt.imshow(cv2.cvtColor(item['imagen'], cv2.COLOR_BGR2RGB))
#            plt.title(item['nombre'])
#            plt.axis('off')
#            
#        plt.tight_layout()
#        plt.show()


# =============================================================================
# --- PARTE B: Solución Definitiva (Black-Hat + Engrosamiento Gris) -----------
# =============================================================================

def segmentar_caracteres(roi_color, nombre_debug="", visualizar=False):
    if roi_color is None or roi_color.size == 0:
        return []

    # --- 1. PREPROCESAMIENTO ---
    h, w = roi_color.shape[:2]
    my = int(h * 0.12)
    mx = int(w * 0.04)
    roi = roi_color[my:h-my, mx:w-mx]
    if roi.size == 0: return []

    # Escala de grises
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --- 2. REALCE DE DETALLES (Black-Hat) ---
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel_bh)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # --- 3. ENGROSAMIENTO EN GRISES ---
    # Variable que queremos graficar
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    blackhat_dilatada = cv2.dilate(blackhat, kernel_dilate, iterations=1)

    # --- 4. BINARIZACIÓN ---
    _, binaria = cv2.threshold(blackhat_dilatada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpieza de bordes
    h_r, w_r = binaria.shape
    cv2.rectangle(binaria, (0,0), (w_r, h_r), 0, 2)

    # --- 5. ANÁLISIS DE CONTORNOS ---
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- VISUALIZACIÓN EXTRA: Contornos Raw ---
    # Dibujamos TODOS los contornos encontrados antes de filtrar
    img_contornos_raw = roi.copy()
    cv2.drawContours(img_contornos_raw, contornos, -1, (0, 0, 255), 1) # Rojo

    candidatos = []
    area_total = h_r * w_r
    
    if visualizar: img_debug = roi.copy()

    for cnt in contornos:
        x, y, w_c, h_c = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # --- FILTROS GEOMÉTRICOS ---
        if h_c < 0.35 * h_r: continue
        if area < 0.015 * area_total: continue

        ratio = w_c / float(h_c)
        if ratio > 6.0: continue 

        # LÓGICA DE SEPARACIÓN
        if ratio > 0.85:
            n_cortes = max(2, int(round(ratio / 0.45)))
            ancho_corte = w_c // n_cortes
            
            for k in range(n_cortes):
                nx = x + k*ancho_corte
                nw = min(ancho_corte, (x + w_c) - nx)
                candidatos.append((nx, roi[y:y+h_c, nx:nx+nw]))
                if visualizar: cv2.rectangle(img_debug, (nx, y), (nx+nw, y+h_c), (0, 255, 255), 2)
        else:
            candidatos.append((x, roi[y:y+h_c, x:x+w_c]))
            if visualizar: cv2.rectangle(img_debug, (x, y), (x+w_c, y+h_c), (0, 255, 0), 2)

    candidatos.sort(key=lambda c: c[0])
    caracteres_finales = [c[1] for c in candidatos]

    # --- DEBUG ACTUALIZADO ---
    if visualizar:
        # Aumentamos el tamaño de la figura para que entren 6 gráficos cómodos
        plt.figure(figsize=(12, 6)) 
        plt.suptitle(f"Proceso Detallado: {nombre_debug}", fontsize=14)

        # 1. Gris original
        plt.subplot(2, 3, 1)
        plt.imshow(gris, cmap='gray')
        plt.title("1. Gris")
        plt.axis('off')

        # 2. Black-Hat (letras extraídas)
        plt.subplot(2, 3, 2)
        plt.imshow(blackhat, cmap='gray')
        plt.title("2. Black-Hat")
        plt.axis('off')

        # 3. Dilatación (NUEVO)
        plt.subplot(2, 3, 3)
        plt.imshow(blackhat_dilatada, cmap='gray')
        plt.title("3. Gris Dilatado (Engrosado)")
        plt.axis('off')

        # 4. Binaria (Otsu)
        plt.subplot(2, 3, 4)
        plt.imshow(binaria, cmap='gray')
        plt.title("4. Binaria (+Otsu)")
        plt.axis('off')

        # 5. Contornos Raw (NUEVO)
        plt.subplot(2, 3, 5)
        # Convertimos a RGB para ver el rojo
        plt.imshow(cv2.cvtColor(img_contornos_raw, cv2.COLOR_BGR2RGB))
        plt.title(f"5. Contornos Raw ({len(contornos)})")
        plt.axis('off')

        # 6. Finales Filtrados
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
        plt.title(f"6. Finales ({len(caracteres_finales)})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return caracteres_finales

# =============================================================================
# --- MAIN ---
# =============================================================================
if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [12, 6]
    lista_patentes = []

    print("Iniciando procesamiento...")

    for i in range(1, 13):
        nombre_archivo = f'img{i:02d}.png'
        
        recorte_patente = obtener_recorte_sin_padding(nombre_archivo, visualizar=False) 
        
        if recorte_patente is not None:
            print(f"-> Analizando {nombre_archivo}...")
            # Visualizar=True para confirmar
            caracteres = segmentar_caracteres(recorte_patente, nombre_debug=nombre_archivo, visualizar=True)
            
            lista_patentes.append({
                "nombre": nombre_archivo, 
                "imagen_patente": recorte_patente,
                "caracteres": caracteres
            })
        else:
            print(f"[{nombre_archivo}] FALLO en Parte A.")

    if lista_patentes:
        filas = len(lista_patentes)
        plt.figure(figsize=(12, filas * 1.5))
        plt.suptitle("RESULTADOS FINALES", fontsize=16)
        
        for i, item in enumerate(lista_patentes):
            plt.subplot(filas, 8, i * 8 + 1)
            plt.imshow(cv2.cvtColor(item['imagen_patente'], cv2.COLOR_BGR2RGB))
            plt.ylabel(item['nombre'], rotation=0, labelpad=40, va='center', fontsize=9)
            plt.xticks([]); plt.yticks([])
            
            chars = item['caracteres']
            for j in range(min(len(chars), 7)): 
                plt.subplot(filas, 8, i * 8 + 2 + j)
                plt.imshow(cv2.cvtColor(chars[j], cv2.COLOR_BGR2RGB))
                plt.axis('off')
        plt.show()