import cv2
import numpy as np
import matplotlib.pyplot as plt

def verificar_transiciones(imagen_binaria_roi):
    """
    Verifica si una región binarizada posee la textura esperada de una patente (texto).
    """
    h, w = imagen_binaria_roi.shape
    if h == 0 or w == 0: return False
    linea_central = imagen_binaria_roi[h // 2, :]
    transiciones = 0
    for i in range(len(linea_central) - 1):
        if linea_central[i] != linea_central[i+1]:
            transiciones += 1
    return 3 <= transiciones <= 50

def obtener_recorte_sin_padding(imagen_path, visualizar=False):
    """
    Localiza y recorta la placa patente utilizando morfología.
    """
    img = cv2.imread(imagen_path)
    h_img, w_img = img.shape[:2]
    p_top, p_bottom, p_side = 0.25, 0.05, 0.20
    y_ini = int(h_img * p_top)
    y_fin = int(h_img * (1 - p_bottom))
    x_ini = int(w_img * p_side)
    x_fin = int(w_img * (1 - p_side))
    
    roi = img[y_ini:y_fin, x_ini:x_fin]
    if roi.size == 0: return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    morf = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_horizontal)

    contornos, _ = cv2.findContours(morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    area_roi = roi.shape[0] * roi.shape[1]

    if visualizar:
        viz_img = img.copy()
        cv2.rectangle(viz_img, (x_ini, y_ini), (x_fin, y_fin), (255, 0, 0), 2)

    for cnt in contornos:
        area_blob = cv2.contourArea(cnt)
        if area_blob < 500:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        if float(bw) / bh < 1.5:
            continue 

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
            
        if short_side == 0:
            continue
        ratio = long_side / short_side
        area = long_side * short_side
        angle_deg = abs(np.degrees(np.arctan2(vec[1], vec[0]))) % 180
        angle_horiz = min(angle_deg, abs(180 - angle_deg))
        if area > (area_roi * 0.2):
            continue 
        if ratio < 1.5 or ratio > 8.0:
            continue 
        if angle_horiz > 45:
            continue 

        mask = np.zeros_like(clean)
        cv2.drawContours(mask, [box], 0, 255, -1)
        val_medio = cv2.mean(clean, mask=mask)[0] / 255.0
        if 0.15 < val_medio < 0.95:
             roi_thresh = thresh[by:by+bh, bx:bx+bw]
             if verificar_transiciones(roi_thresh):
                 rect_global = ((rect[0][0] + x_ini, rect[0][1] + y_ini), (rect[1][0], rect[1][1]), rect[2])
                 candidatos.append((rect_global, ratio))

    mejor_candidato = None
    mejor_score = 1000
    for cand in candidatos:
        r_struct, r_ratio = cand
        diff = abs(r_ratio - 3.2)
        if diff < mejor_score:
            mejor_score = diff
            mejor_candidato = r_struct

    roi_patente = None
    if mejor_candidato:
        box = np.int32(cv2.boxPoints(mejor_candidato))
        if visualizar:
            cv2.drawContours(viz_img, [box], 0, (0, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(box)
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        if w > 0 and h > 0:
            roi_patente = img[y:y+h, x:x+w]

    if visualizar:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.title("Detección"); plt.imshow(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2); plt.title("Recorte"); 
        if roi_patente is not None:
            plt.imshow(cv2.cvtColor(roi_patente, cv2.COLOR_BGR2RGB))
        else:
            plt.text(0.5, 0.5, "No detectado", ha='center')
        plt.show()

    return roi_patente

lista_patentes = []
for i in range(1, 13):
    nombre_archivo = f'img{i:02d}.png'
    recorte = obtener_recorte_sin_padding(nombre_archivo, visualizar=False) #Setear visualizar en True para mostrar pasos
    if recorte is not None:
        lista_patentes.append({"nombre": nombre_archivo, "imagen": recorte})
    else:
        print(f"{nombre_archivo}: No detectada.")

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