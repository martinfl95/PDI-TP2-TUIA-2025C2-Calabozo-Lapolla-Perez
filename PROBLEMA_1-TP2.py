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
    img_salida = img.copy()
    monedas_count = 0
    dados_count = 0
    contornos_monedas = []
    contornos_dados = []
    print('-'*50)
    print("SEGMENTACION POR TIPO")
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
        #como para implementar los valores teóricos proporcionados
        if metrica > 15.5: 
            dados_count += 1
            color = (0, 0, 255) # Rojo
            etiqueta = f"D {metrica:.1f}"
            print(f"{'DADO':<10} | {int(area):<8} | {metrica:.4f}")
            contornos_dados.append({'id': i, 'dado': cnt, 'area': area, 'centro': (cY,cX), 'metrica': metrica})
            cv2.putText(img_salida, etiqueta, (cX - 400, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        else:
            monedas_count += 1
            color = (0, 255, 0) # Verde
            etiqueta = f"M {metrica:.1f}"
            print(f"{'MONEDA':<10} | {int(area):<8} | {metrica:.4f}")
            contornos_monedas.append({'id': i, 'moneda': cnt, 'area': area, 'centro': (cY,cX), 'metrica': metrica})
            cv2.putText(img_salida, etiqueta, (cX - 400, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.drawContours(img_salida, [cnt], -1, color, 3)
        
    if 3 in etapas:
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(img_salida, cv2.COLOR_BGR2RGB))
        plt.title(f"ETAPA 3: Detección - {monedas_count} Monedas - {dados_count} Dados - Factor de Forma: 15.5", fontsize=16)
        plt.axis('off')
        plt.show()
    
    return contornos_monedas, contornos_dados

def normalizar_contornos(monedas_detectadas: list, imagen: str= 'monedas.jpg', graficar: bool = False):
    monedas_elipses = []
    
    img = cv2.imread(imagen)
    if img is None: 
        print("Error: No se pudo cargar la imagen.")
        return
    
    img_salida = None
    if graficar:
        img_salida = img.copy()
            
    for item in monedas_detectadas:
            cnt = item['moneda']
            #Elipse de area mínima segun nuestro contorno
            elipse = cv2.fitEllipse(cnt)
            monedas_elipses.append(elipse)
            if graficar and img_salida is not None:
                #Elipse calculada
                cv2.ellipse(img_salida, elipse, (0, 255, 0), 3)
                centro = (int(elipse[0][0]), int(elipse[0][1]))
                #Centro de la elipse
                cv2.circle(img_salida, centro, 5, (0, 0, 255), -1)

    if graficar and img_salida is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(img_salida, cv2.COLOR_BGR2RGB))
        plt.title("ETAPA 4 - Normalización - Elipses Ajustadas al Contorno Original")
        plt.axis('off')
        plt.show()
        
    return monedas_elipses

def clasificar_monedas(elipses_monedas: list, imagen: str = 'monedas.jpg', graficar: bool = False):
    img = plt.imread(imagen)
    img_salida = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = []
    
    #Umbral de corte para monedas 
    corte_10_c = 74000
    corte_50_c = 94000
    
    print('-'*50)
    print("CLASIFICACION DE MONEDAS")
    print(f"{'ID':<5} | {'AREA':<10} | {'TIPO':<15}")
    print("-" * 50)

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
        
        #Guardamos el resultado de la clasificación
        resultados.append({
            'id': i,
            'area': area_elipse,
            'clasificacion': etiqueta,
            'centro': center
        })
        
        #Print de resultados para encontrar umbrales de separación
        print(f"{i:<5} | {int(area_elipse):<10} | {etiqueta}")
        
        #Gráficamos las elipses y etiquetas sobre la imagen de salida
        if graficar and img_salida is not None:
            cv2.ellipse(img_salida, elipse, color_elipse, 2)
            cv2.putText(img_salida, etiqueta, (center[0]-300, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color_elipse, 4, cv2.LINE_AA)

    if graficar and img_salida is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(img_salida, cv2.COLOR_BGR2RGB))
        plt.title("ETAPA 5 - Clasificación por Tamaño (Area de la Elipse)")
        plt.axis('off')
        plt.show()
        
    return resultados

def clasificar_dados(dados_detectados: list, imagen: str ='monedas.jpg', graficar: bool = True):
    
    img = cv2.imread(imagen) 
    if img is None: 
        print("Error: No se pudo cargar la imagen.")
        return []    
    
    img_salida = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_grises = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resultados = []
    
    # Gráficos intermedios
    if graficar:
        cant_dados = len(dados_detectados)
        if cant_dados > 0:
            fig, axes = plt.subplots(nrows=cant_dados, ncols=4, figsize=(20, 5 * cant_dados))
            if cant_dados == 1: axes = np.array([axes])
        else:
            print("No se detectaron dados para graficar.")

    #Encabezados del print
    print('-'*50)
    print("CLASIFICACION DE DADOS")
    print(f"{'ID':<5} | {'VALOR':<10}")
    print("-" * 50)
    
    
    for i, item in enumerate(dados_detectados):
        cnt = item['dado']
        id_dado = item.get('id', 0)
        
        # Bounding box asociado al contorno
        x, y, w, h = cv2.boundingRect(cnt)
        
        #Area de interés sobre la imagen en escala de grises
        roi_gris = img_grises[y:y+h, x:x+w].copy()
        
        #Máscara del tamaño del bounding box
        mascara = np.zeros((h, w), dtype=np.uint8)
        #Ajuste de coordenadas globales a locales
        cnt_local = cnt - np.array([x, y])
        cv2.drawContours(mascara, [cnt_local], -1, 255, -1)
        
        #Máscara de visualización, fondo negro
        roi_enmascarada_visual = roi_gris.copy()
        roi_enmascarada_visual[mascara == 0] = 0 
        
        #Máscara fondo blanco para facilitar encontrar los contornos
        roi_gris[mascara == 0] = 255 
        
        # Filtros para suavizado y umbralado
        roi_blur = cv2.GaussianBlur(roi_gris, (5, 5), 0)
        _, roi_umbralado = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        #Contornos internos de los dados
        puntos, _ = cv2.findContours(roi_umbralado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valor_dado = 0
        #Conversión a RGB para graficar
        roi_umbralado_color = cv2.cvtColor(roi_umbralado, cv2.COLOR_GRAY2RGB)
        
        for p in puntos:
            area_p = cv2.contourArea(p)
            if area_p == 0: continue
            #Perímetro y factor de forma inverso
            perimetro_p = cv2.arcLength(p, True)
            metrica = (perimetro_p ** 2) / area_p
            #Filtro para los puntos que quedaron contenidos dentro de la máscara que no forman parte de la cara principal
            es_circular = (metrica < 15.0)
            
            if es_circular:
                valor_dado += 1
                if graficar:
                    #Volvemos a las coordenadas globales
                    p_global = p + np.array([x, y])
                    cv2.drawContours(img_salida, [p_global], -1, (0, 255, 0), 2)
                    cv2.drawContours(roi_umbralado_color, [p], -1, (0, 255, 0), -1) 
                         
        item['valor'] = valor_dado
        #Agregamos el valor del dado al diccionario obtenido de la segmentación
        resultados.append(item)
        
        print(f"{id_dado:<5} | {valor_dado:<10}")

        if graficar:
            cY,cX = item['centro']
            #Texto con el valor encontrado para el dado a tratar
            cv2.putText(img_salida, str(f'Valor: {valor_dado}'), (cX-500, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

            # Subplots para mostrar los pasos
            axes[i, 0].imshow(img_grises[y:y+h, x:x+w], cmap='gray')
            axes[i, 0].set_title(f"Dado {id_dado}: ROI Original")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(roi_enmascarada_visual, cmap='gray')
            axes[i, 1].set_title("Máscara")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(roi_umbralado_color)
            axes[i, 2].set_title("Umbralado + Filtro")
            axes[i, 2].axis('off')

            pad = 20
            y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
            x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
            axes[i, 3].imshow(img_salida[y1:y2, x1:x2])
            axes[i, 3].set_title(f"Resultado: {valor_dado}")
            axes[i, 3].axis('off')

    if graficar:
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 12))
        plt.imshow(img_salida)
        plt.title("ETAPA 6 - Clasificación de Dados por Factor de Forma")
        plt.axis('off')
        plt.show()
        
    return resultados

if __name__ == '__main__':
    '''
    etapas: ([1,2,3] | []) - corresponde a los gráficos de cada paso específico. Ej: etapas = [1,2] grafica el procedimiento
    de las primeras dos etapas
    
    graficar_monedas (True | False): desarrollo de la etapa de normalización
    
    graficar_clasificacion_monedas (True | False): desarrollo de la etapa de clasificación
    
    graficar_clasificacion_dados (True | False): desarrollo de la etapa de clasificación de dados
    
    histograma_factor_forma (True | False): Histograma de umbral de decisión de dados y monedas
    
    histograma_areas (True | False): Histograma de umbral de decisión de tipos de monedas
    
    En la terminal, luego de la ejecución, habrá resúmenes sobre los valores calculados y encontrados en cada etapa
    '''
    #Etapa 1: Preprocesamiento - filtro mediana sobre imagen original para suavizar bordes
    #Etapa 2: Morfología - Tratamiento de imagen para segmentar contornos
    #Etapa 3: Clasificación - Utilizando factor de forma encontramos dados o monedas
    #Etapa 4: Normalización de contornos - Encontramos una elipse de minimo tamaño que regularice los contornos encontrados
    #Etapa 5: Clasificación Monedas - Utilizando el area de cada moneda diferenciamos los tipos
    #Etapa 6: Clasificacion Dados - Utilizando contornos internos y factor de forma encontramos el valor de la cara
    etapas = []
    graficar_monedas = False
    graficar_clasificacion_monedas = False
    graficar_clasificacion_dados = True
    histograma_factor_forma = False
    histograma_areas = False
    contornos_monedas, contornos_dados = segmentacion_contornos('monedas.jpg', etapas = etapas)
    if contornos_monedas:
        contornos_monedas_normalizados = normalizar_contornos(contornos_monedas, 'monedas.jpg', graficar_monedas)
    if contornos_monedas_normalizados:
        resultados_monedas = clasificar_monedas(contornos_monedas_normalizados, 'monedas.jpg', graficar_clasificacion_monedas)
    if contornos_dados:
        resultados_dados = clasificar_dados(contornos_dados, 'monedas.jpg', graficar_clasificacion_dados)
        
    if histograma_factor_forma:
        valores_monedas = [obj['metrica'] for obj in contornos_monedas]
        valores_dados = [obj['metrica'] for obj in contornos_dados]

        plt.figure(figsize=(10, 6))
        
        plt.hist(valores_monedas, bins=10, alpha=0.7, label='Monedas', color='skyblue', edgecolor='black')
        plt.hist(valores_dados, bins=10, alpha=0.7, label='Dados', color='salmon', edgecolor='black')
        if valores_monedas and valores_dados:
            umbral_sugerido = (max(valores_monedas) + min(valores_dados)) / 2
            plt.axvline(umbral_sugerido, color='red', linestyle='dashed', linewidth=2, label=f'Umbral (~{umbral_sugerido:.2f})')

        plt.xlabel('Factor de Forma Inverso ($P^2/A$)', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.title('Distribución del Factor de Forma: Separabilidad de Clases', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
    if histograma_areas:
        areas = [m['area'] for m in resultados_monedas]
        corte_10_c = 74000
        corte_50_c = 94000
        plt.figure(figsize=(10, 6))
        plt.hist(areas, bins=15, color='gray', edgecolor='black', alpha=0.7, label='Distribución de Áreas')
        plt.axvline(corte_10_c, color='red', linestyle='--', linewidth=2, label=f'Corte 10c ({corte_10_c})')
        plt.axvline(corte_50_c, color='blue', linestyle='--', linewidth=2, label=f'Corte 50c ({corte_50_c})')
        plt.xlabel('Área (píxeles cuadrados)', fontsize=12)
        plt.ylabel('Frecuencia (Cantidad de monedas)', fontsize=12)
        plt.title('Histograma de Áreas y Umbrales de Clasificación', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()