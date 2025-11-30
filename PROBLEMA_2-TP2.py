import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def verificar_transiciones(imagen_binaria_roi):
    h, w = imagen_binaria_roi.shape
    if h == 0 or w == 0: return False
    linea_central = imagen_binaria_roi[h // 2, :]
    transiciones = 0
    for i in range(len(linea_central) - 1):
        if linea_central[i] != linea_central[i+1]:
            transiciones += 1
    return 3 <= transiciones <= 50

def obtener_recorte_v13(imagen_path, visualizar=False):
    """
    Lógica V13: Morfología Direccional (Limpieza Vertical + Unión Horizontal).
    Retorna: El recorte de la patente (imagen color) o None si falla.
    """
    if not os.path.exists(imagen_path): return None
    img = cv2.imread(imagen_path)
    if img is None: return None

    # --- 1. ROI (Configuración V13) ---
    h_img, w_img = img.shape[:2]
    # 25% arriba, 5% abajo, 20% lados
    p_top, p_bottom, p_side = 0.25, 0.05, 0.20
    y_ini = int(h_img * p_top)
    y_fin = int(h_img * (1 - p_bottom))
    x_ini = int(w_img * p_side)
    x_fin = int(w_img * (1 - p_side))
    
    roi = img[y_ini:y_fin, x_ini:x_fin]
    if roi.size == 0: return None

    # --- 2. PREPROCESAMIENTO ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 3. BORDES ---
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 4. MORFOLOGÍA DIRECCIONAL (La clave de V13) ---
    # A. Limpieza Vertical (Mata adoquines)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)
    
    # B. Conexión Horizontal (Une letras)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    morf = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_horizontal)

    # --- 5. ANÁLISIS ---
    contornos, _ = cv2.findContours(morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    area_roi = roi.shape[0] * roi.shape[1]

    if visualizar:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x_ini, y_ini), (x_fin, y_fin), (255, 0, 0), 2)

    for cnt in contornos:
        # Filtros rápidos
        area_blob = cv2.contourArea(cnt)
        if area_blob < 500: continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0: continue
        if float(bw) / bh < 1.5: continue 

        # Análisis Fino
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        
        d1 = np.linalg.norm(box[0] - box[1])
        d2 = np.linalg.norm(box[1] - box[2])
        
        if d1 > d2:
            long_side, short_side = d1, d2
            vec = box[0] - box[1]
        else:
            long_side, short_side = d2, d1
            vec = box[1] - box[2]
            
        if short_side == 0: continue
        
        ratio = long_side / short_side
        area = long_side * short_side
        
        # Ángulo
        angle_deg = abs(np.degrees(np.arctan2(vec[1], vec[0]))) % 180
        angle_horiz = min(angle_deg, abs(180 - angle_deg))

        # Filtros V13 estrictos
        if area > (area_roi * 0.2): continue 
        if ratio < 1.5 or ratio > 8.0: continue 
        if angle_horiz > 45: continue 

        # Contenido
        mask = np.zeros_like(clean)
        cv2.drawContours(mask, [box], 0, 255, -1)
        val_medio = cv2.mean(clean, mask=mask)[0] / 255.0
        
        if 0.15 < val_medio < 0.95:
             roi_thresh = thresh[by:by+bh, bx:bx+bw]
             if verificar_transiciones(roi_thresh):
                 # Guardar ajustando coordenadas al original
                 rect_global = ((rect[0][0] + x_ini, rect[0][1] + y_ini), (rect[1][0], rect[1][1]), rect[2])
                 candidatos.append((rect_global, ratio))

    # --- 6. SELECCIÓN ---
    mejor_candidato = None
    mejor_score = 1000
    
    for cand in candidatos:
        r_struct, r_ratio = cand
        diff = abs(r_ratio - 3.2)
        if diff < mejor_score:
            mejor_score = diff
            mejor_candidato = r_struct

    # --- 7. EXTRACCIÓN Y RETORNO ---
    roi_patente = None
    
    if mejor_candidato:
        box = np.int32(cv2.boxPoints(mejor_candidato))
        
        # Visualización en Debug
        if visualizar:
            cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 3)

        # Recorte Seguro (Axis Aligned) con un pequeño margen
        x, y, w, h = cv2.boundingRect(box)
        
        # Padding del 10% para no cortar bordes
        pad_w = int(w * 0.10)
        pad_h = int(h * 0.10)
        
        x_s = max(0, x - pad_w)
        y_s = max(0, y - pad_h)
        w_s = min(w + 2*pad_w, img.shape[1] - x_s)
        h_s = min(h + 2*pad_h, img.shape[0] - y_s)
        
        if w_s > 0 and h_s > 0:
            roi_patente = img[y_s:y_s+h_s, x_s:x_s+w_s]

    # Plot opcional
    if visualizar:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.title("Detección V13"); plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2); plt.title("Recorte Retornado"); 
        if roi_patente is not None: plt.imshow(cv2.cvtColor(roi_patente, cv2.COLOR_BGR2RGB))
        else: plt.text(0.5, 0.5, "No detectado", ha='center')
        plt.show()

    return roi_patente

# =============================================================================
# BLOQUE PRINCIPAL: Procesamiento de Lote y Guardado
# =============================================================================

lista_patentes = [] # Esta es la estructura que usarás en el Apartado B

print("Iniciando segmentación con lógica V13...")

for i in range(1, 13):
    nombre_archivo = f'img{i:02d}.png'
    
    # Llamada a la función. Poner visualizar=True si quieres ver una por una.
    recorte = obtener_recorte_v13(nombre_archivo, visualizar=False)
    
    if recorte is not None:
        print(f"[OK] {nombre_archivo}: Guardada.")
        # Guardamos un diccionario con nombre y la imagen (array)
        lista_patentes.append({
            "nombre": nombre_archivo,
            "imagen": recorte
        })
    else:
        print(f"[X]  {nombre_archivo}: No se detectó patente.")

print(f"\nProcesamiento completo. {len(lista_patentes)}")

# Visualización rápida de lo que guardaste en la lista
if len(lista_patentes) > 0:
    cols = 4
    rows = (len(lista_patentes) // cols) + 1
    plt.figure(figsize=(15, rows * 3))
    
    for idx, item in enumerate(lista_patentes):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(cv2.cvtColor(item['imagen'], cv2.COLOR_BGR2RGB))
        plt.title(item['nombre'])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(15, 10))
cols = 4
rows = (len(lista_patentes) // cols) + 1

for i, item in enumerate(lista_patentes):
    # Accedemos a la IMAGEN dentro del diccionario
    imagen_bgr = item['imagen'] 
    
    # Convertimos a RGB para que matplotlib muestre los colores bien
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    plt.subplot(rows, cols, i + 1)
    plt.imshow(imagen_rgb)
    plt.title(item['nombre'])
    plt.axis('off')

plt.tight_layout()
plt.show()