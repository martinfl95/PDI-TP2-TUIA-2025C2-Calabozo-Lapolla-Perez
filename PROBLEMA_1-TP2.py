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

def segmentacion_final_visual(imagen_path):
    img = cv2.imread(imagen_path)
    if img is None: 
        print("Error: No se pudo cargar la imagen.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    edges = cv2.Canny(blur, 20, 120)

    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,19))
    edges_thick = cv2.dilate(edges, kernel_dilatacion, iterations=2)
    
    mask_filled = imfillhole(edges_thick)
    
    mask_eroded = cv2.erode(mask_filled, kernel_dilatacion, iterations=2)
    
    mask_final = cv2.medianBlur(mask_eroded, 13)
    plt.figure(figsize=(16, 10))
    
    plt.subplot(231)
    plt.imshow(edges, cmap='gray')
    plt.title("1. Canny (Bordes)")
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(edges_thick, cmap='gray')
    plt.title("2. Dilatación (Anillos)")
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(mask_filled, cmap='gray')
    plt.title("3. Relleno (Discos)")
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(mask_eroded, cmap='gray')
    plt.title("4. Erosión (Tamaño Real)")
    plt.axis('off')
    
    plt.subplot(235)
    plt.imshow(mask_final, cmap='gray')
    plt.title("5. Final (Suavizado k=13)")
    plt.axis('off')

    contours_final, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_out = img.copy()
    monedas_count = 0
    dados_count = 0
    
    print(f"{'TIPO':<10} | {'AREA':<8} | {'METRICA (P^2/A)':<20}")
    print("-" * 50)

    for cnt in contours_final:
        area = cv2.contourArea(cnt)
        if area < 1000: continue 
        
        perimetro = cv2.arcLength(cnt, True)
        if area == 0: continue
        
        metrica = (perimetro ** 2) / area
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else: cX, cY = 0, 0

        if metrica > 15.3: 
            dados_count += 1
            color = (0, 0, 255) # Rojo
            etiqueta = f"D {metrica:.1f}"
            print(f"{'DADO':<10} | {int(area):<8} | {metrica:.4f}")
        else:
            monedas_count += 1
            color = (0, 255, 0) # Verde
            etiqueta = f"M {metrica:.1f}"
            print(f"{'MONEDA':<10} | {int(area):<8} | {metrica:.4f}")
            
        cv2.drawContours(img_out, [cnt], -1, color, 3)
        cv2.putText(img_out, etiqueta, (cX - 30, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    plt.subplot(236)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(f"6. RESULTADO: {monedas_count} Monedas | {dados_count} Dados")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

segmentacion_final_visual('monedas.jpg')