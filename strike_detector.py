import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado con tu dataset
model = YOLO("yolov8n.pt")  # cambia a tu ruta real

# Abrir webcam o archivo de video
cap = cv2.VideoCapture(0)  # Cambia a "video.mp4" si quieres usar un archivo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar predicción
    results = model(frame)[0]

    # Variables para saber si hay zona y pelota
    strike_zone_box = None
    ball_box = None

    for r in results.boxes:
        cls = int(r.cls[0])
        label = model.names[cls]
        box = r.xyxy[0].cpu().numpy().astype(int)

        if label == "strike_zone":
            strike_zone_box = box
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
            cv2.putText(frame, "Strike Zone", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif label == "ball":
            ball_box = box
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 2)
            cv2.putText(frame, "Ball", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Lógica para determinar si es strike
    if strike_zone_box is not None and ball_box is not None:
        x1, y1, x2, y2 = strike_zone_box
        bx1, by1, bx2, by2 = ball_box

        # Verificar si el centro de la pelota está dentro de la zona
        ball_center_x = (bx1 + bx2) / 2
        ball_center_y = (by1 + by2) / 2

        if x1 < ball_center_x < x2 and y1 < ball_center_y < y2:
            cv2.putText(frame, "STRIKE!", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "BALL!", (50, 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Strike Detection", frame)

    if cv2.waitKey(1) == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()
