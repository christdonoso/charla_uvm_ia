import cv2
import mediapipe as mp
import numpy as np


def calcular_angulo(a,b,c):
    """Calcula el angulo entre 3 articulaciones en coordenadas
    de un plano cartesiano

    Args:
        a tuple: coordenadas articulacion proximal
        b tuple: coordenadas articulacion intermedia
        c tuple: _coordenadas articulacion distal

    Returns:
        floar: medida del angulo aproximada a 2 cifras
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian* 180 /np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle,2)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Variables de control
threshold = 90  # Umbral del ángulo
below_threshold = False  # Flag para detectar el cruce del umbral
rep_count = 0  # Contador de repeticiones

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_height, img_width, _ = frame.shape
    
    #obtener los landmarks del video
    results = pose.process(frame)
    try:
        print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER])
        #rescatamos los puntos de interes
        shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        body = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y] 
        
        angle = calcular_angulo(body, shoulder, elbow)
        
    except:
        angle = 0
        shoulder = [0,0]
        elbow = [0,0]
        wrist = [0,0]
    
    mp_drawing.draw_landmarks(
        frame, 
        results.pose_landmarks, 
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=2)
    )

    cv2.putText(
        frame,
        str(round(angle, 2)),
        tuple(map(int,([shoulder[0] * img_width, shoulder[1] * img_height ]))),  # Convertimos a enteros
        #tuple(map(int, np.multiply(elbow, [640, 480])))
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    if angle < threshold:
        below_threshold = True  # El ángulo bajó del umbral
    elif below_threshold and angle >= threshold:
        rep_count += 1  # Se cuenta una repetición cuando el ángulo sube nuevamente
        below_threshold = False  # Se reinicia el flag para detectar la siguiente repetición

    cv2.putText(
        frame,
        str(rep_count),
        tuple(map(int,([50, img_height - 50 ]))),  # Convertimos a enteros
        #tuple(map(int, np.multiply(elbow, [640, 480])))
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        3,
        cv2.LINE_AA
    )

    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
