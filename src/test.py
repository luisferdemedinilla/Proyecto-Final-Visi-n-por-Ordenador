
import cv2
import numpy as np

# Parámetros obtenidos de la calibración
camera_matrix = np.array([
    [775.3770952, 0, 367.08087401],
     [0, 777.943537795, 510.10299332],
     [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array([1.97599071e-01,  -1.24057799e+00, -3.06253123e-04 ,-3.08980276e-03, 2.83269649e+00], dtype=np.float64)


def main(camera_index=1, width=1280, height=720):

    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Corregir la distorsión de la lente
            corrected = cv2.undistort(
                frame,
                camera_matrix,
                dist_coeffs
            )

            # Voltear ambas imágenes (efecto espejo)
            frame = cv2.flip(frame, 1)
            corrected = cv2.flip(corrected, 1)

            # Mostrar ambas imágenes
            cv2.imshow("Camara original", frame)
            cv2.imshow("Camara corregida", corrected)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=1, width=1280, height=720)
