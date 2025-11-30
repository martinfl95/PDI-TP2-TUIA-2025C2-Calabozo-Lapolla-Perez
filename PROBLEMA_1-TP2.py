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
        plt.title("2. Dilate (21x21)")
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(mascara_rellenada, cmap='gray')
        plt.title("3. Relleno")
        plt.axis('off')
        
        plt.subplot(234)
        plt.imshow(mascara_erosion, cmap='gray')
        plt.title("4. Erosion (21x21)")
        plt.axis('off')
        
        plt.subplot(236)
        plt.imshow(mascara_final, cmap='gray')
        plt.title("5. Filtro de Mediana (13x13)")
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
    contornos_monedas = []
    contornos_dados = []
    
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
        
        #Encontramos el centro de los contornos
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else: cX, cY = 0, 0
        
        #Tuvimos que modificar la métrica para que captara correctamente las monedas y los dados
        #No podemos encontrar una combinación que nos dé perfectamente el contorno de cada elemento
        #como para implementar los valores teóricos
        if metrica > 15.5: 
            dados_count += 1
            color = (0, 0, 255) # Rojo
            etiqueta = f"D {metrica:.1f}"
            print(f"{'DADO':<10} | {int(area):<8} | {metrica:.4f}")
            contornos_dados.append({'id': i, 'dado': cnt, 'area': area, 'centro': (cY,cX), 'metrica': metrica})
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

def normalizar_contornos(contorno_monedas: list, imagen: str= 'monedas.jpg', graficar: bool = False):
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
            #Elipse de area mínima segun nuestro contorno
            elipse = cv2.fitEllipse(cnt)
            monedas_elipses.append(elipse)
            if graficar and img_out is not None:
                #Elipse calculada
                cv2.ellipse(img_out, elipse, (0, 255, 0), 3)
                center = (int(elipse[0][0]), int(elipse[0][1]))
                #Centro de la elipse
                cv2.circle(img_out, center, 5, (0, 0, 255), -1)

    if graficar and img_out is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title("Normalización: Elipses Ajustadas al Contorno original")
        plt.axis('off')
        plt.show()
        
    return monedas_elipses

def clasificar_monedas(elipses_monedas: list, imagen: str = 'monedas.jpg', graficar: bool = False):
    img = plt.imread(imagen)
    img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = [] 
    corte_10_c = 74000
    corte_50_c = 94000

    print(f"{'ID':<5} | {'AREA':<10} | {'TIPO':<15}")
    print("-" * 40)

    for i, elipse in enumerate(elipses_monedas):
        (cX, cY), (w, h), angle = elipse
        #Cálculo de area de elipse Pi*DiagMen*DiagMay/4
        area_elipse = (np.pi * w * h) / 4
        center = (int(cX), int(cY))
        
        if area_elipse < corte_10_c:
            etiqueta = "10C" 
            color_elipse = (0, 255, 255)
            
        elif area_elipse < corte_50_c:
            etiqueta = "1P"
            color_elipse = (255, 0, 0) 
            
        else:
            etiqueta = "50C"
            color_elipse = (0, 0, 255)
            
        resultados.append({
            'id': i,
            'area': area_elipse,
            'clasificacion': etiqueta,
            'centro': center
        })
        
        #Print de resultados para encontrar umbrales de separación
        print(f"{i:<5} | {int(area_elipse):<10} | {etiqueta}")
        
        if graficar and img_out is not None:
            cv2.ellipse(img_out, elipse, color_elipse, 2)
            cv2.putText(img_out, etiqueta, (center[0]-300, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_elipse, 4, cv2.LINE_AA)

    if graficar and img_out is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.title("Clasificación por Tamaño (Area de la Elipse)")
        plt.axis('off')
        plt.show()
        
    return resultados

def clasificar_dados(dados_detectados: list, imagen, graficar: bool = True):
    pass

if __name__ == '__main__':
    '''
    etapas: ([1,2,3] | []) - corresponde a los gráficos de cada paso específico. Ej: etapas = [1,2] grafica el procedimiento
    de las primeras dos etapas
    
    graficar_monedas (True | False): desarrollo de la etapa de normalización
    
    graficar_clasificacion_monedas (True | False): desarrollo de la etapa de clasificación
    
    graficar_dados (True | False): desarrollo de la etapa de clasificación de dados
    '''
    #Etapa 1: Preprocesamiento - filtro mediana sobre imagen original para suavizar bordes
    #Etapa 2: Morfología - Tratamiento de imagen para segmentar contornos
    #Etapa 3: Clasificación - Utilizando factor de forma encontramos dados o monedas
    #Etapa 4: Normalización de contornos - Encontramos una elipse de minimo tamaño que regularice los contornos encontrados
    #Etapa 5: Clasificación Monedas - Utilizando el area de cada moneda diferenciamos los tipos
    #Etapa 6: Clasificacion Dados - Utilizando contornos internos y factor de forma encontramos el valor de la cara
    etapas = [1,2,3]
    graficar_monedas = True
    graficar_clasificacion_monedas = True
    contornos_monedas, contornos_dados = segmentacion_contornos('monedas.jpg', etapas = etapas)
    contornos_monedas_normalizados = normalizar_contornos(contornos_monedas, 'monedas.jpg', graficar_monedas)
    resultados_monedas = clasificar_monedas(contornos_monedas_normalizados, 'monedas.jpg', graficar_clasificacion_monedas)
    resultados_dados = clasificar_dados(contornos_dados, 'moneda.jpg', True)