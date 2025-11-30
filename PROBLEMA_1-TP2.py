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

def segmentacion_contornos(imagen, etapas = []):
    img = cv2.imread(imagen)
    if img is None: 
        print("Error: No se pudo cargar la imagen.")
        return

    # ---------------------------------------------------------
    # Etapa 1: Preprocesamiento
    # ---------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    # GRAFICO 1: Original + Blur
    if 1 in etapas:
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
    # Etapa 2 : Segmentación y Morfología
    # ---------------------------------------------------------
    # 1. Canny
    bordes = cv2.Canny(blur, 45, 145)
    
    # 2. Dilatación
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    bordes_gruesos = cv2.dilate(bordes, kernel_dilatacion, iterations=2)
    
    # 3. Relleno
    mascara_rellenada = imfillhole(bordes_gruesos)
    
    # 4. Erosión (Vuelta al tamaño real)
    mascara_erosion = cv2.erode(mascara_rellenada, kernel_dilatacion, iterations=2)
    
    # 5. Filtrado de ruido final
    mascara_final = cv2.medianBlur(mascara_erosion, 13)

    # GRAFICO 2: Pipeline Morfológico
    if 2 in etapas:
        plt.figure(figsize=(16, 5))
        plt.suptitle("ETAPA 2: Segmentación y Relleno", fontsize=16)

        plt.subplot(231)
        plt.imshow(bordes, cmap='gray')
        plt.title("1. Canny")
        plt.axis('off')

        plt.subplot(232)
        plt.imshow(bordes_gruesos, cmap='gray')
        plt.title("3. Dilate (21x21)")
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(mascara_rellenada, cmap='gray')
        plt.title("4. Relleno")
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(mascara_erosion, cmap='gray')
        plt.title("5. Erosion (21x21)")
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(mascara_final, cmap='gray')
        plt.title("6. Filtro de Mediana (13x13)")
        plt.axis('off')
        plt.show()
    # ---------------------------------------------------------
    # Etapa 3:  Clasificación y Resultado
    # ---------------------------------------------------------
    contours_final, _ = cv2.findContours(mascara_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_monedas = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
    mascara_dados = np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8)
    img_out = img.copy()
    monedas_count = 0
    dados_count = 0
    contornos_monedas = [{}]
    contornos_dados =[{}]
    
    print(f"{'TIPO':<10} | {'AREA':<8} | {'METRICA (P^2/A)':<20}")
    print("-" * 50)

    for i, cnt in enumerate(contours_final):
        area = cv2.contourArea(cnt)
        #Eliminamos contornos que no posean un area factible
        if area < 1000: continue 
        #Calculamos el perímetro
        perimetro = cv2.arcLength(cnt, True)
        if area == 0: continue
        #Calculamos el factor de forma
        metrica = (perimetro ** 2) / area
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else: cX, cY = 0, 0
        #Tuvimos que modificar la métrica para que captara correctamente las monedas y los dados
        #No podemos encontrar una combinación que nos dé perfectamente el contorno de cada elemento
        #como para implementar el mínimo y máximos teóricos
        if metrica > 15.5: 
            dados_count += 1
            color = (0, 0, 255) # Rojo
            etiqueta = f"D {metrica:.1f}"
            print(f"{'DADO':<10} | {int(area):<8} | {metrica:.4f}")
            contornos_monedas.append({'id': i, 'dado': cnt, 'area': area, 'centro': (cY,cX), 'metrica': metrica})
            cv2.putText(img_out, etiqueta, (cX - 400, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        else:
            monedas_count += 1
            color = (0, 255, 0) # Verde
            etiqueta = f"M {metrica:.1f}"
            print(f"{'MONEDA':<10} | {int(area):<8} | {metrica:.4f}")
            contornos_monedas.append({'id': i, 'moneda': cnt, 'area': area, 'centro': (cY,cX), 'metrica': metrica})
            cv2.putText(img_out, etiqueta, (cX - 400, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.drawContours(img_out, [cnt], -1, color, 3)
        
    if 3 in etapas:
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title(f"ETAPA 3: Detección - {monedas_count} Monedas - {dados_count} Dados - Factor de Forma: 15.5", fontsize=16)
        plt.axis('off')
        plt.show()
    
    return contornos_monedas, contornos_dados

def normalizar_contornos(contorno_monedas: list, imagen, graficar: bool = False):
    monedas_elipses = []
    img_out = None
    if graficar:
            temp_img = cv2.imread(imagen)
            if temp_img is None:
                print("Error: No se pudo cargar la imagen para graficar.")
                return []
            img_out = temp_img.copy()
            
    for item in contorno_monedas:
            cnt = item['moneda']
            elipse = cv2.fitEllipse(cnt)
            monedas_elipses.append(elipse)
            if graficar and img_out is not None:
                cv2.ellipse(img_out, elipse, (0, 255, 0), 3)
                center = (int(elipse[0][0]), int(elipse[0][1]))
                cv2.circle(img_out, center, 5, (0, 0, 255), -1)

    if graficar and img_out is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title(f"Normalización: {len(monedas_elipses)} Elipses Ajustadas")
        plt.axis('off')
        plt.show()
        
    return monedas_elipses
#Para observar los pasos intermedios pueden agregar el identificador de etapas
#Pseudocódigo -
#Etapa 1: Preprocesamiento - filtro mediana sobre imagen original para suavizar bordes
#Etapa 2: Morfología - Tratamiento de imagen para segmentar contornos
#Etapa 3: Clasificación - Utilizando factor de forma encontramos dados o monedas
#Etapa 4: Clasificación Monedas - Utilizando color y area diferenciamos los tipos
#Etapa 5: Clasificacion Dados - Utilizando contornos internos
contornos_monedas, contornos_dados = segmentacion_contornos('monedas.jpg') #Etapa 1, 2 y 3
contornos_monedas_normalizados = normalizar_contornos(contornos_monedas, 'monedas.jpg', True) #Etapa 3.1