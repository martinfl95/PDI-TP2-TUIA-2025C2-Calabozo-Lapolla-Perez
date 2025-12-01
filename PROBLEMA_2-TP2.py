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

# ===================== PARTE B – Reconocimiento de caracteres en patentes =====================
# IMPORTANTE: este bloque asume que ya existe la lista `lista_patentes`
# generada en la PARTE A del TP (NO se modifica nada de la parte A).

# --------------------- B.1 – Normalizar chapa dentro del ROI --------------------- #

def normalizar_chapa(roi_bgr, debug=False, nombre=""):
    """
    A partir del recorte grande (salida de la parte A),
    intenta quedarse solo con la chapa (rectángulo negro con letras blancas).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    # Paso a gris + suavizado + realce de contraste
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    # Umbralado Otsu invertido: zona oscura de chapa → blanco
    _, bw = cv2.threshold(gray_eq, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morfología: queremos que toda la chapa sea un bloque
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
    bw_close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    bw_clean = cv2.morphologyEx(bw_close, cv2.MORPH_OPEN, k_open)

    # Componentes conectadas: buscamos la región que parezca más a una chapa
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw_clean, connectivity=8
    )

    h, w = bw_clean.shape
    area_img = h * w

    mejor_idx = None
    mejor_score = -1e9

    for i in range(1, num_labels):  # salteo fondo
        x, y, ww, hh, area = stats[i]

        # descarto objetos demasiado chicos o enormes
        if area < 0.01 * area_img or area > 0.5 * area_img:
            continue

        aspect = ww / float(hh) if hh > 0 else 0
        # chapa aprox 3:1 (permitimos variación por perspectiva)
        if aspect < 2.0 or aspect > 6.0:
            continue

        # score = grande + aspecto cercano a 3
        score = area - 2000 * abs(aspect - 3.0)

        if score > mejor_score:
            mejor_score = score
            mejor_idx = i

    if mejor_idx is None:
        # si no hay nada razonable, devuelvo el ROI original
        chapa = roi_bgr.copy()
    else:
        x, y, ww, hh, area = stats[mejor_idx]
        # margen pequeño
        mx = int(0.05 * ww)
        my = int(0.15 * hh)
        x0 = max(x - mx, 0)
        y0 = max(y - my, 0)
        x1 = min(x + ww + mx, w)
        y1 = min(y + hh + my, h)
        chapa = roi_bgr[y0:y1, x0:x1].copy()

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        ax[0].set_title(f"ROI {nombre}")
        ax[0].imshow(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")

        ax[1].set_title("BW limpio")
        ax[1].imshow(bw_clean, cmap="gray")
        ax[1].axis("off")

        ax[2].set_title("Chapa normalizada")
        ax[2].imshow(cv2.cvtColor(chapa, cv2.COLOR_BGR2RGB))
        ax[2].axis("off")
        plt.tight_layout()
        plt.show()

    return chapa


# Normalizamos todas las chapas a partir de lista_patentes (parte A)
lista_chapas = []
for item in lista_patentes:
    nombre = item["nombre"]
    roi = item["imagen"]

    chapa = normalizar_chapa(roi, debug=False, nombre=nombre)
    lista_chapas.append({
        "nombre": nombre,
        "imagen": chapa
    })


# --------------------- B.2 – Binarizar chapa y segmentar caracteres --------------------- #

def binarizar_chapa_para_caracteres(chapa_bgr):
    """
    Devuelve una imagen binaria donde idealmente
    las letras/dígitos aparecen en blanco (255) sobre fondo negro.
    """
    gray = cv2.cvtColor(chapa_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    # fondo negro, caracteres blancos
    _, bw = cv2.threshold(gray_eq, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # erosión suave para adelgazar contornos
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw_er = cv2.erode(bw, k, iterations=1)

    return bw_er


def segmentar_caracteres(chapa_bgr, debug=False, nombre=""):
    """
    A partir de la imagen de la chapa (color),
    devuelve:
      - lista de recortes de caracteres (imágenes binarias)
      - lista de cajas (x, y, w, h) relativas a la chapa
    """
    bw = binarizar_chapa_para_caracteres(chapa_bgr)
    h, w = bw.shape

    # Recorte vertical del renglón de caracteres (proyección horizontal)
    proy_h = np.sum(bw == 255, axis=1)
    filas = np.where(proy_h > 0)[0]
    if len(filas) == 0:
        return [], []

    y0 = max(filas[0] - 2, 0)
    y1 = min(filas[-1] + 2, h)
    banda = bw[y0:y1, :]

    bh, bw_ancho = banda.shape
    area_banda = bh * bw_ancho

    # Componentes conectadas en la banda
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        banda, connectivity=8
    )

    candidatos = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]

        # filtro por área: ni ruido, ni toda la banda
        if area < 0.005 * area_banda or area > 0.2 * area_banda:
            continue

        # proporción geométrica típica de caracteres
        aspect = hh / float(ww) if ww > 0 else 0
        if aspect < 1.0 or aspect > 4.5:
            continue

        # altura relativa: al menos 40% de la banda
        if hh < 0.4 * bh:
            continue

        candidatos.append((x, y, ww, hh, area))

    if len(candidatos) == 0:
        return [], []

    # Tomo los 6 más grandes y ordeno de izquierda a derecha
    candidatos = sorted(candidatos, key=lambda c: c[4], reverse=True)
    candidatos = candidatos[:6]
    candidatos = sorted(candidatos, key=lambda c: c[0])

    recortes = []
    cajas = []
    for (x, y, ww, hh, area) in candidatos:
        char_img = banda[y:y+hh, x:x+ww]
        recortes.append(char_img)
        cajas.append((x, y + y0, ww, hh))  # coordenadas relativas a la chapa completa

    if debug:
        vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        for (x, y, ww, hh) in [(c[0], c[1] + y0, c[2], c[3]) for c in candidatos]:
            cv2.rectangle(vis, (x, y), (x+ww, y+hh), (0, 0, 255), 1)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].set_title(f"Chapa {nombre}")
        ax[0].imshow(cv2.cvtColor(chapa_bgr, cv2.COLOR_BGR2RGB)); ax[0].axis("off")

        ax[1].set_title("Binaria")
        ax[1].imshow(bw, cmap="gray"); ax[1].axis("off")

        ax[2].set_title("Caracteres detectados")
        ax[2].imshow(vis); ax[2].axis("off")
        plt.tight_layout()
        plt.show()

    return recortes, cajas


# --------------------- B.3 – Modelos de caracteres (plantillas) --------------------- #

def cargar_modelos_caracteres(carpeta, size=(30, 50)):
    """
    Lee imágenes de la carpeta y arma un diccionario:
      modelos['A'] = [lista de imágenes binarias normalizadas de 'A']
    Los archivos deben llamarse por ejemplo: A_1.png, B_1.png, 0_1.png, etc.
    """
    modelos = {}
    if not os.path.isdir(carpeta):
        print(f"[ADVERTENCIA] Carpeta de modelos no encontrada: {carpeta}")
        return modelos

    for fname in os.listdir(carpeta):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        ruta = os.path.join(carpeta, fname)
        etiqueta = fname.split("_")[0].upper()  # 'A', 'B', '0', '1', etc.

        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        _, bw = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        modelos.setdefault(etiqueta, []).append(bw)

    print(f"[INFO] Modelos cargados: {list(modelos.keys())}")
    return modelos


def similitud_binaria(a, b):
    """
    a y b: imágenes binarias 0/255 del mismo tamaño.
    Métrica simple tipo '1 - promedio del XOR'.
    """
    a = (a > 0).astype(np.float32)
    b = (b > 0).astype(np.float32)
    diff = np.abs(a - b)
    return 1.0 - diff.mean()


def reconocer_un_caracter(char_img, modelos, size=(30, 50)):
    """
    Devuelve (mejor_etiqueta, mejor_score) para el caracter dado.
    """
    char_res = cv2.resize(char_img, size, interpolation=cv2.INTER_AREA)
    _, char_bw = cv2.threshold(char_res, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mejor_label = "?"
    mejor_score = -1.0

    for etiqueta, lista in modelos.items():
        for tmpl in lista:
            s = similitud_binaria(char_bw, tmpl)
            if s > mejor_score:
                mejor_score = s
                mejor_label = etiqueta

    return mejor_label, mejor_score


# --------------------- B.4 – Reconocer todas las patentes --------------------- #

# Ruta a la carpeta donde tengas tus ejemplos de caracteres (A_1.png, 0_1.png, etc.)
carpeta_modelos = "modelos_caracteres"   # CAMBIAR SI USÁS OTRA RUTA
modelos = cargar_modelos_caracteres(carpeta_modelos)

resultados_patentes = []

for item in lista_chapas:
    nombre = item["nombre"]
    chapa = item["imagen"]

    chars, cajas = segmentar_caracteres(chapa, debug=False, nombre=nombre)

    texto = ""
    scores = []

    for ch in chars:
        label, sc = reconocer_un_caracter(ch, modelos)
        texto += label
        scores.append(sc)

    # formateo tipo XXX 999 si hay 6 caracteres
    if len(texto) == 6:
        texto_fmt = texto[:3] + " " + texto[3:]
    else:
        texto_fmt = texto

    resultados_patentes.append((nombre, texto_fmt, scores))
    print(f"{nombre}: {texto_fmt}  -> scores {np.round(scores, 3)}")

# (Opcional) si querés guardar en un diccionario por nombre:
dict_resultados_patentes = {nombre: texto for (nombre, texto, _) in resultados_patentes}
