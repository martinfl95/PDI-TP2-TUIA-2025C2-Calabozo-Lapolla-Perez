import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    
    Parámetros:
        ruta_imagen (str): Ruta del archivo.
        mostrar_pasos (bool): Si es True, muestra una figura con las 6 etapas del procesamiento.
    """
    imagen = cv2.imread(ruta_imagen)
    if imagen is None: return None
    
    alto_img, ancho_img = imagen.shape[:2]

    # ROI
    p_arriba, p_abajo, p_costado = 0.25, 0.05, 0.20
    y_ini = int(alto_img * p_arriba)
    y_fin = int(alto_img * (1 - p_abajo))
    x_ini = int(ancho_img * p_costado)
    x_fin = int(ancho_img * (1 - p_costado))
    roi = imagen[y_ini:y_fin, x_ini:x_fin]
    if roi.size == 0: return None

    # Preprocesamiento
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gris_eq = clahe.apply(gris)
    desenfoque = cv2.GaussianBlur(gris_eq, (5, 5), 0)

    # Bordes (Sobel)
    sobelx = cv2.Sobel(desenfoque, cv2.CV_64F, 1, 0, ksize=3)
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    
    # Umbralado (Otsu)
    _, umbralizada = cv2.threshold(sobelx_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfología
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    limpia = cv2.morphologyEx(umbralizada, cv2.MORPH_OPEN, kernel_vertical)
    
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    morfologia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel_horizontal)

    # Análisis de contornos
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

    # Extración
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



if __name__ == '__main__':
    lista_patentes = []

    for i in range(1, 13):
        nombre_archivo = f'img{i:02d}.png'
        
        # recorte limpio de la patente
        recorte = obtener_recorte(nombre_archivo, mostrar_pasos=True) 
        
        if recorte is not None:
            lista_patentes.append({"nombre": nombre_archivo, "imagen": recorte})
        else:
            print(f"[{nombre_archivo}] No detectada.")

    # Visualización de las patentes juntas
    if lista_patentes:
        cols = 4
        filas = (len(lista_patentes) // cols) + 1
        plt.figure(figsize=(15, filas * 3))
        
        for idx, item in enumerate(lista_patentes):
            plt.subplot(filas, cols, idx + 1)
            plt.imshow(cv2.cvtColor(item['imagen'], cv2.COLOR_BGR2RGB))
            plt.title(item['nombre'])
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()