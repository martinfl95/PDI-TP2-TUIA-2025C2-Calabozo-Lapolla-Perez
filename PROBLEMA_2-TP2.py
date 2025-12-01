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

def obtener_recorte(imagen_path, mostrar_pasos=False):
    """
    Localiza y recorta la placa patente utilizando morfología.
    
    Parámetros:
        mostrar_pasos (bool): Si es True, muestra una figura con las etapas intermedias.
    """
    img = cv2.imread(imagen_path)
    h_img, w_img = img.shape[:2]
    p_arriba, p_abajo, p_lados = 0.25, 0.05, 0.20
    y_ini = int(h_img * p_arriba)
    y_fin = int(h_img * (1 - p_abajo))
    x_ini = int(w_img * p_lados)
    x_fin = int(w_img * (1 - p_lados))
    roi = img[y_ini:y_fin, x_ini:x_fin]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize = 3)
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    _, thresh = cv2.threshold(sobelx_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
    for cnt in contornos:
        # 1. Filtro de Área Mínima: Descartar ruido pequeño
        area_blob = cv2.contourArea(cnt)
        if area_blob < 500: continue

        # 2. Filtro de Proporción del Bounding Box recto
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0: continue
        if float(bw) / bh < 1.5: continue 

        rect = cv2.minAreaRect(cnt)
        box_points = np.int32(cv2.boxPoints(rect))
        
        d1 = np.linalg.norm(box_points[0] - box_points[1])
        d2 = np.linalg.norm(box_points[1] - box_points[2])
        
        if d1 > d2:
            long_side, short_side = d1, d2
            vec = box_points[0] - box_points[1]
        else:
            long_side, short_side = d2, d1
            vec = box_points[1] - box_points[2]
            
        if short_side == 0: continue
        
        ratio = long_side / short_side
        area = long_side * short_side
        
        angle_deg = abs(np.degrees(np.arctan2(vec[1], vec[0]))) % 180
        angle_horiz = min(angle_deg, abs(180 - angle_deg))
        
        if area > (area_roi * 0.2): continue 
        if ratio < 1.5 or ratio > 8.0: continue 
        if angle_horiz > 45: continue 

        mask = np.zeros_like(clean)
        cv2.drawContours(mask, [box_points], 0, 255, -1)
        val_medio = cv2.mean(clean, mask = mask)[0] / 255.0
        
        if 0.15 < val_medio < 0.95:
             roi_thresh = thresh[by : by + bh, bx : bx + bw]
             if verificar_transiciones(roi_thresh):
                 rect_global = ((rect[0][0] + x_ini, rect[0][1] + y_ini), (rect[1][0], rect[1][1]), rect[2])
                 candidatos.append((rect_global, ratio))

    # Selección de candidatos
    # Buscamos el candidato cuyo ratio se acerque más a los estándares (2.0 o 3.1)
    mejor_candidato = None
    mejor_score = float('inf')
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
        box = np.int32(cv2.boxPoints(mejor_candidato))

        x, y, w, h = cv2.boundingRect(box)
        pad_w = int(w * 0.10)
        pad_h = int(h * 0.10)
        x_s = max(0, x - pad_w)
        y_s = max(0, y - pad_h)
        w_s = min(w + 2*pad_w, img.shape[1] - x_s)
        h_s = min(h + 2*pad_h, img.shape[0] - y_s)
        
        if w_s > 0 and h_s > 0:
            roi_patente = img[y_s:y_s+h_s, x_s:x_s+w_s]

    if mostrar_pasos:
        plt.figure(figsize=(16, 8))
        plt.subplot(2, 3, 1); plt.imshow(gray, cmap='gray'); plt.title("1. ROI Grises")
        plt.subplot(2, 3, 2); plt.imshow(sobelx_abs, cmap='gray'); plt.title("2. Sobel Vertical")
        plt.subplot(2, 3, 3); plt.imshow(thresh, cmap='gray'); plt.title("3. Umbralado (Otsu)")
        plt.subplot(2, 3, 4); plt.imshow(clean, cmap='gray'); plt.title("4. Limpieza Vertical")
        plt.subplot(2, 3, 5); plt.imshow(morf, cmap='gray'); plt.title("5. Unión Horizontal")
        plt.subplot(2, 3, 6)
        if roi_patente is not None:
            plt.imshow(cv2.cvtColor(roi_patente, cv2.COLOR_BGR2RGB))
            plt.title("6. Resultado")
        else:
            plt.text(0.5, 0.5, "No detectado", ha='center')
            plt.title("6. Resultado")
            
        plt.tight_layout()
        plt.show()

    return roi_patente

lista_patentes = []
for i in range(1, 13):
    nombre_archivo = f'img{i:02d}.png'
    #mostrar_pasos=True muestra los pasos en el procesamiento
    recorte = obtener_recorte(nombre_archivo, mostrar_pasos=True) 
    if recorte is not None:
        lista_patentes.append({"nombre": nombre_archivo, "imagen": recorte})
    else:
        print(f"[X] {nombre_archivo}: No se detectó patente.")

cols = 4
rows = (len(lista_patentes) // cols) + 1
plt.figure(figsize=(15, rows * 3))
for idx, item in enumerate(lista_patentes):
    imagen_rgb = cv2.cvtColor(item['imagen'], cv2.COLOR_BGR2RGB)
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(imagen_rgb)
    plt.title(item['nombre'])
    plt.axis('off')
plt.tight_layout()
plt.show()