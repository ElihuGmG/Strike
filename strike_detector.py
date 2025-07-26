import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime

# Configuración de color para pelota blanca
color_bajo = np.array([0, 0, 180])
color_alto = np.array([180, 50, 255])

escala_metros_por_pixel = 1 / 250  # Ajusta según tu escena real
fps = 30
camera_index = 0

# Inicialización
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FPS, fps)

# Archivo CSV
if not os.path.exists('registro.csv'):
    with open('registro.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fecha', 'Velocidad (km/h)'])

# Seguimiento
posicion_anterior = None
tiempo_anterior = time.time()

# Botones
boton_salir = [(10, 420), (130, 460)]
boton_cambiar = [(150, 420), (320, 460)]

def dibujar_boton(frame, esquina1, esquina2, texto, color=(50, 50, 50)):
    cv2.rectangle(frame, esquina1, esquina2, color, -1)
    cv2.putText(frame, texto, (esquina1[0] + 10, esquina2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def dentro_del_boton(pos, esquina1, esquina2):
    return esquina1[0] <= pos[0] <= esquina2[0] and esquina1[1] <= pos[1] <= esquina2[1]

def cambiar_camara():
    global camera_index, cap
    camera_index += 1
    cap.release()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)

# Mouse callback
def manejar_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if dentro_del_boton((x, y), *boton_salir):
            print("Saliendo...")
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        elif dentro_del_boton((x, y), *boton_cambiar):
            print("Cambiando cámara...")
            cambiar_camara()

cv2.namedWindow("Strike - Medidor de velocidad")
cv2.setMouseCallback("Strike - Medidor de velocidad", manejar_click)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, color_bajo, color_alto)

    # Ruido fuera
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_DILATE, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    velocidad_kmh = 0
    estado = "🔍 Buscando pelota..."

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if 300 < area < 1500:
            (x, y, w, h) = cv2.boundingRect(contorno)
            aspect_ratio = float(w) / h

            if 0.8 < aspect_ratio < 1.2:
                perimeter = cv2.arcLength(contorno, True)
                if perimeter == 0:
                    continue
                circularidad = 4 * np.pi * (area / (perimeter ** 2))

                if circularidad > 0.7:
                    centro = (int(x + w / 2), int(y + h / 2))
                    cv2.circle(frame, centro, 8, (0, 255, 0), -1)

                    tiempo_actual = time.time()
                    if posicion_anterior is not None:
                        distancia_pixeles = np.linalg.norm(np.array(centro) - np.array(posicion_anterior))
                        dt = tiempo_actual - tiempo_anterior

                        if dt > 0:
                            velocidad_m_s = (distancia_pixeles * escala_metros_por_pixel) / dt
                            velocidad_kmh = velocidad_m_s * 3.6
                            estado = "✅ Pelota detectada"

                            with open('registro.csv', 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), round(velocidad_kmh, 2)])

                    posicion_anterior = centro
                    tiempo_anterior = tiempo_actual
                    break

    # Fondo oscuro transparente para texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 80), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Texto visual
    cv2.putText(frame, f"Velocidad: {velocidad_kmh:.2f} km/h", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(frame, estado, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    # Dibujar botones
    dibujar_boton(frame, *boton_salir, "Salir", (60, 0, 0))
    dibujar_boton(frame, *boton_cambiar, "Cambiar cámara", (0, 60, 0))

    cv2.imshow("Strike - Medidor de velocidad", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
