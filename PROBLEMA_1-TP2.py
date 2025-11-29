import cv2
import numpy as np
import matplotlib.pyplot as plt

def segmentar_monedas_dados_por_forma(imagen_path, mostrar=True):
    img = cv2.imread(imagen_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    monedas = []
    dados = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        factor_forma = (4 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if factor_forma > 0.8:
            monedas.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif 0.5 < factor_forma < 0.8 and 0.8 < aspect_ratio < 1.2:
            dados.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if mostrar:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Monedas: {len(monedas)} | Dados: {len(dados)}")
        plt.axis("off")
        plt.show()
    return monedas, dados

monedas, dados = segmentar_monedas_dados_por_forma('monedas.jpg')
print(monedas)
print(dados)