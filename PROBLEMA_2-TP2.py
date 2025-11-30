import cv2
import numpy as np
import matplotlib.pyplot as plt

def clear_border(img_bin):
    """
    Elimina componentes conectados que tocan el borde de la imagen
    usando reconstrucción morfológica por dilatación.
    img_bin: imagen binaria 0/255 con objetos en blanco.
    """
    img = img_bin.copy().astype(np.uint8)
    h, w = img.shape[:2]

    # Marcador: sólo los pixeles de borde
    marker = np.zeros_like(img, dtype=np.uint8)
    marker[0, :]   = img[0, :]
    marker[-1, :]  = img[-1, :]
    marker[:, 0]   = img[:, 0]
    marker[:, -1]  = img[:, -1]

    kernel = np.ones((3, 3), np.uint8)

    prev = np.zeros_like(img, dtype=np.uint8)
    while True:
        marker = cv2.dilate(marker, kernel)
        marker = cv2.bitwise_and(marker, img)
        if np.array_equal(marker, prev):
            break
        prev = marker.copy()

    # Resto esos objetos de la imagen original
    img_cb = cv2.subtract(img, marker)
    return img_cb

# ------------------------------------------------------------------
# FUNCIÓN A: DETECCIÓN Y SEGMENTACIÓN DE LA PATENTE (PUNTO A)
# Basada en Sobel + Otsu + Morfología (Unidad 2, 3 y 6)
# ------------------------------------------------------------------
def segmentar_patente_robusta(imagen_path, mostrar=True):
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen {imagen_path}")
        return []

    # Paso 1: gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Paso 2: normalización de iluminación (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Paso 3: suavizado (Gaussiano)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Paso 4: Sobel en x (bordes verticales)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    # Paso 5: Umbralado (Otsu)
    _, thresh = cv2.threshold(
        sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Paso 6: Clausura morfológica para unir bordes verticales
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morf = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, elemento)

    # Paso 7: Contornos candidatos a placa
    contornos, _ = cv2.findContours(
        morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    patentes_candidatas = []
    debug_img = img.copy()

    # Ordenamos contornos por área (mayores primero)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:15]

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h

        # Filtros geométricos relajados para placa
        if 1.5 < aspect_ratio < 6.0 and area > 300:
            # Densidad de pixeles blancos dentro del ROI binario
            roi_bin = thresh[y:y+h, x:x+w]
            white_pixels = cv2.countNonZero(roi_bin)
            density = white_pixels / (w * h + 1e-6)

            if 0.2 < density < 0.8:
                # Pequeño padding para no cortar caracteres
                pad_w = int(w * 0.05)
                pad_h = int(h * 0.15)
                x_cut = max(0, x - pad_w)
                y_cut = max(0, y - pad_h)
                w_cut = min(img.shape[1] - x_cut, w + 2 * pad_w)
                h_cut = min(img.shape[0] - y_cut, h + 2 * pad_h)

                roi_color = img[y_cut:y_cut+h_cut, x_cut:x_cut+w_cut]

                # Chequeo tamaño lógico
                if roi_color.shape[0] > 10 and roi_color.shape[1] > 10:
                    patentes_candidatas.append(roi_color)
                    cv2.rectangle(
                        debug_img, (x_cut, y_cut),
                        (x_cut + w_cut, y_cut + h_cut),
                        (0, 255, 0), 2
                    )

    if mostrar:
        plt.figure(figsize=(13, 4))
        plt.subplot(1, 4, 1)
        plt.title("Gris + CLAHE")
        plt.imshow(gray_eq, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Sobel X")
        plt.imshow(sobelx, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Otsu + Clausura")
        plt.imshow(morf, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title(f"Patentes detectadas: {len(patentes_candidatas)}")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.suptitle(f"Detección de placa - {imagen_path}")
        plt.tight_layout()
        plt.show()

    return patentes_candidatas


# ------------------------------------------------------------------
# FUNCIÓN B: SEGMENTACIÓN DE CARACTERES (PUNTO B)
# Usa umbralado, morfología y descriptores geométricos.
# ------------------------------------------------------------------
def segmentar_caracteres(placa_roi, mostrar=True, titulo=""):
    """
    Segmenta caracteres de la placa usando:
    - Ecualización / filtro
    - Top-hat (resalta caracteres claros sobre fondo oscuro)
    - Umbralado de Otsu
    - Apertura + clausura morfológica
    - Eliminación de objetos que tocan el borde
    - Componentes conectados + filtrado por área y aspecto
    """
    # --- 1) Paso a grises y mejora de contraste -------------------------
    gray = cv2.cvtColor(placa_roi, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_fil = cv2.medianBlur(gray_eq, 3)

    # --- 2) Top-hat morfológico (resalta letras blancas sobre fondo negro)
    h, w = gray_fil.shape
    k = max(9, (h // 3) | 1)   # tamaño impar ~ altura de caracteres
    se_th = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    th_img = cv2.morphologyEx(gray_fil, cv2.MORPH_TOPHAT, se_th)

    # --- 3) Umbralado (Otsu) + invertido: caracteres en blanco ---------
    _, bw = cv2.threshold(th_img, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw

    # --- 4) Limpieza morfológica (abertura + clausura) -----------------
    se_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, se_small)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se_small)

    # --- 5) Elimino objetos que tocan el borde (marco de la patente) ----
    bw_cb = clear_border(bw)

    # --- 6) Componentes conectados -------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw_cb, connectivity=8
    )

    chars = []
    boxes = []

    area_img = h * w
    for i in range(1, num_labels):   # label 0 = fondo
        x, y, wc, hc, area = stats[i]

        # Filtros heurísticos para quedarnos con caracteres
        if area < 0.01 * area_img:
            continue                 # demasiado chico, ruido
        if hc < 0.4 * h or hc > 0.95 * h:
            continue                 # muy bajo o muy alto
        aspect = wc / float(hc)
        if aspect < 0.25 or aspect > 0.9:
            continue                 # muy finito o muy ancho

        char = bw_cb[y:y+hc, x:x+wc]
        boxes.append((x, y, wc, hc))
        chars.append(char)

    # Ordeno de izquierda a derecha
    boxes_chars = sorted(zip(boxes, chars), key=lambda p: p[0][0])
    chars = [c for (_, c) in boxes_chars]

    # --- 7) Visualización opcional -------------------------------------
    if mostrar:
        cols = max(3, len(chars) + 2)
        plt.figure(figsize=(3 * cols, 4))

        ax1 = plt.subplot(1, cols, 1)
        plt.imshow(cv2.cvtColor(placa_roi, cv2.COLOR_BGR2RGB))
        plt.title("ROI Patente")
        plt.axis("off")

        plt.subplot(1, cols, 2, sharey=ax1)
        plt.imshow(bw_cb, cmap='gray')
        plt.title("Binaria (inv) limpia")
        plt.axis("off")

        for i, ch in enumerate(chars):
            plt.subplot(1, cols, 3 + i)
            plt.imshow(ch, cmap='gray')
            plt.title(f"Char {i+1}")
            plt.axis("off")

        plt.suptitle(f"Segmentación de caracteres {titulo}")
        plt.tight_layout()
        plt.show()

    return chars




# ------------------------------------------------------------------
# PROCESAMIENTO DE LAS 12 IMÁGENES
# ------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(1, 13):
        imagen = f'img{i:02d}.png'
        print(f"\n======================")
        print(f"Procesando {imagen}")
        print(f"======================")

        patentes = segmentar_patente_robusta(imagen, mostrar=True)

        if not patentes:
            print(f"No se detectó ninguna patente en {imagen}")
            continue

        for idx, placa_roi in enumerate(patentes):
            titulo = f"{imagen} - Patente {idx+1}"
            caracteres = segmentar_caracteres(placa_roi, mostrar=True, titulo=titulo)
            print(f"{titulo}: {len(caracteres)} caracteres detectados")
#