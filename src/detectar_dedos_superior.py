"""Detecta las puntas de los dedos y muestra sus coordenadas en pixeles.

Necesita hand_landmarker.task junto a este archivo.
correcion debe recibir una imagen BGR y devolver la imagen BGR corregida.
Ajusta el import siguiente al nombre del archivo que contiene tu funcion.
"""
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np



CAMARA = 0  # Prueba 1 si se abre otra webcam.
INVERTIR_MANOS = False  # Activalo si las etiquetas izquierda/derecha salen invertidas.
MODELO = Path(__file__).with_name("hand_landmarker.task")

DEDOS = {
    4: "Pulgar",
    8: "Indice",
    12: "Corazon",
    16: "Anular",
    20: "Menique",
}

def correcion(frame):
    """Devuelve la imagen BGR corregida de distorsion.
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    return frame
"""
def main():
    if not MODELO.is_file():
        raise FileNotFoundError(f"Coloca hand_landmarker.task junto a este archivo: {MODELO}")

    opciones = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=MODELO.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    nombres_manos = {"Left": "Izquierda", "Right": "Derecha"}
    if INVERTIR_MANOS:
        nombres_manos = {"Left": "Derecha", "Right": "Izquierda"}

    camara = cv2.VideoCapture(CAMARA)
    ultimo_instante = -1
    try:
        if not camara.isOpened():
            raise RuntimeError(f"No se pudo abrir la camara {CAMARA}.")

        with mp.tasks.vision.HandLandmarker.create_from_options(opciones) as detector:
            print("SI")
            while True:
                ok, frame = camara.read()
                if not ok:
                    raise RuntimeError("No se pudo leer una imagen de la camara.")

                # Corregir la distorsion antes de calcular las posiciones.
                #frame = correcion(frame)
                if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError("correcion(frame) debe devolver una imagen BGR de 3 canales.")

                alto, ancho = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                instante = max(time.monotonic_ns() // 1_000_000, ultimo_instante + 1)
                ultimo_instante = instante
                resultado = detector.detect_for_video(imagen, instante)

                # Panel lateral con las coordenadas de las yemas.
                panel = np.zeros((max(alto, 480), 360, 3), dtype=np.uint8)
                cv2.putText(panel, "Posiciones (x, y) en pixeles", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                fila = 60
                for puntos, etiquetas in zip(resultado.hand_landmarks, resultado.handedness):
                    mano = nombres_manos[etiquetas[0].category_name]
                    color = (0, 220, 0) if mano == "Izquierda" else (0, 180, 255)
                    cv2.putText(panel, mano, (10, fila),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                    fila += 28

                    for indice, dedo in DEDOS.items():
                        punto = puntos[indice]
                        x = round(punto.x * (ancho - 1))
                        y = round(punto.y * (alto - 1))
                        cv2.circle(frame, (x, y), 6, color, -1)
                        cv2.putText(frame, f"{mano[0]}-{dedo}", (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        cv2.putText(panel, f"{dedo}: ({x}, {y})", (15, fila),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                        fila += 26
                    fila += 16

                if not resultado.hand_landmarks:
                    cv2.putText(panel, "No se detectan manos", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                vista = np.zeros((panel.shape[0], ancho + panel.shape[1], 3), dtype=np.uint8)
                vista[:alto, :ancho] = frame
                vista[:, ancho:] = panel
                cv2.imshow("Dedos - Q o Esc para salir", vista)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except Exception as e:
        print("Error:", e)
        

    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
