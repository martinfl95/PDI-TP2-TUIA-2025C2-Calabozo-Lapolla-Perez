import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentar_patente_robusta(imagen_path, mostrar = True):
    img = cv2.imread(imagen_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normaliza la luz
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Suavizado para reducir ruido
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # SOBEL vertical
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    # Binarización
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morfología
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morf = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, elemento)
    contornos, _ = cv2.findContours(morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patentes_candidatas = []
    debug_img = img.copy()
    # Ordenamiento por área, se procesan los más grandes primero
    contornos = sorted(contornos, key = cv2.contourArea, reverse = True)[:15]
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        # FILTROS RELAJADOS
        # Bajamos el área mínima (por si el auto está lejos)
        # Ampliamos el ratio (por si la patente está inclinada)
        if 1.5 < aspect_ratio < 6.0 and area > 300: # Antes era > 800
            # VALIDACIÓN DE DENSIDAD
            roi_bin = thresh[y:y+h, x:x+w]
            white_pixels = cv2.countNonZero(roi_bin)
            density = white_pixels / (w * h)
            # Rango un poco más amplio
            if 0.2 < density < 0.8:
                # RECORTE SEGURO
                # A veces el contorno es muy justo, damos un pequeño margen
                pad_w = int(w * 0.05)
                pad_h = int(h * 0.15)
                x_cut = max(0, x - pad_w)
                y_cut = max(0, y - pad_h)
                w_cut = w + 2*pad_w
                h_cut = h + 2*pad_h
                roi_color = img[y_cut:y_cut+h_cut, x_cut:x_cut+w_cut]              
                # Verificación final de tamaño lógico (evita recortes de 1x1 pixel)
                if roi_color.shape[0] > 10 and roi_color.shape[1] > 10:
                    patentes_candidatas.append(roi_color)
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if mostrar:
        plt.figure(figsize=(12, 4))        
        plt.subplot(1, 3, 1)
        plt.title("Sobel + Otsu")
        plt.imshow(thresh, cmap='gray')
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Morfología")
        plt.imshow(morf, cmap='gray')
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title(f"Detectadas: {len(patentes_candidatas)}")
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    return patentes_candidatas

for i in range(1, 13):
    imagen = f'img{i:02d}.png'
    segmentar_patente_robusta(imagen)