import cv2
import numpy as np
import matplotlib.pyplot as plt

def imreconstruct(marker, mask, kernel=None):
    if kernel is None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   
        if (marker == expanded_intersection).all():                         
            break                                                           
        marker = expanded_intersection        
    return expanded_intersection

def imfillhole(img):
    mask = np.zeros_like(img)                                                   
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) 
    marker = cv2.bitwise_not(img, mask=mask)                
    img_c = cv2.bitwise_not(img)                            
    img_r = imreconstruct(marker=marker, mask=img_c)        
    img_fh = cv2.bitwise_not(img_r)                         
    return img_fh

def segmentacion_por_pasos(imagen_path):
    img = cv2.imread(imagen_path)
    if img is None: 
        print("Error: No se pudo cargar la imagen.")
        return

    # ---------------------------------------------------------
    # Preprocesamiento
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # GRAFICO 1: Original + Blur
    plt.figure(figsize=(12, 6))
    plt.suptitle("ETAPA 1: Preprocesamiento", fontsize=16)
    
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(blur, cmap='gray')
    plt.title("Filtro Mediana (k=5)")
    plt.axis('off')
    plt.show()

    # ---------------------------------------------------------
    # Segmentación y Morfología
    # ---------------------------------------------------------
    # 1. Canny
    bordes = cv2.Canny(blur, 45, 145)

    # 2. Close (Clausura)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Dilatación
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    bordes_gruesos = cv2.dilate(bordes_cerrados, kernel_dilatacion, iterations=2)
    
    # 4. Relleno
    mascara_rellenada = imfillhole(bordes_gruesos)
    
    # 5. Erosión (Vuelta al tamaño real)
    mascara_erosion = cv2.erode(mascara_rellenada, kernel_dilatacion, iterations=2)
    
    # 6. Lijado Final
    mascara_final = cv2.medianBlur(mascara_erosion, 13)

    # GRAFICO 2: Pipeline Morfológico
    plt.figure(figsize=(16, 5))
    plt.suptitle("ETAPA 2: Segmentación y Relleno", fontsize=16)

    plt.subplot(141)
    plt.imshow(bordes, cmap='gray')
    plt.title("1. Canny")
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(bordes_cerrados, cmap='gray')
    plt.title("2. Close (31x31)")
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(bordes_gruesos, cmap='gray')
    plt.title("3. Dilate (21x21)")
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(mascara_rellenada, cmap='gray')
    plt.title("4. Relleno Final")
    plt.axis('off')
    plt.show()

    # ---------------------------------------------------------
    # Clasificación y Resultado
    # ---------------------------------------------------------
    contours_final, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_out = img.copy()
    monedas_count = 0
    dados_count = 0
    
    print(f"{'TIPO':<10} | {'AREA':<8} | {'METRICA (P^2/A)':<20}")
    print("-" * 50)

    for cnt in contours_final:
        area = cv2.contourArea(cnt)
        #Eliminamos contornos que no posean un area factible
        if area < 1000: continue 
        #Calculamos el perímetro...
        perimetro = cv2.arcLength(cnt, True)
        #Suavizamos el perímetro para obtener máscaras con menor ruido
        epsilon = 0.007 * perimetro
        cnt_suave = cv2.approxPolyDP(cnt, epsilon, True)
        perimetro_suave = cv2.arcLength(cnt_suave, True)
        if area == 0: continue
        #Calculamos el factor de forma
        metrica = (perimetro_suave ** 2) / area
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else: cX, cY = 0, 0
        #Tuvimos que modificar la métrica para que captara correctamente las monedas y los dados
        #No podemos encontrar una combinación que nos dé perfectamente el contorno de cada elemento
        #como para implementar el mínimo y máximos teóricos
        if metrica > 13.5: 
            dados_count += 1
            color = (0, 0, 255) # Rojo
            etiqueta = f"D {metrica:.1f}"
            print(f"{'DADO':<10} | {int(area):<8} | {metrica:.4f}")
        else:
            monedas_count += 1
            color = (0, 255, 0) # Verde
            etiqueta = f"M {metrica:.1f}"
            print(f"{'MONEDA':<10} | {int(area):<8} | {metrica:.4f}")
            
        cv2.drawContours(img_out, [cnt_suave], -1, color, 3)
        cv2.putText(img_out, etiqueta, (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(f"ETAPA 3: Resultado Final | {monedas_count} Monedas | {dados_count} Dados", fontsize=16)
    plt.axis('off')
    plt.show()

# Ejecutar
segmentacion_por_pasos('monedas.jpg')