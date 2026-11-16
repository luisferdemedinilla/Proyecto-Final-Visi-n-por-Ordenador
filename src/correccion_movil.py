
import cv2
import numpy as np

# Parámetros obtenidos de la calibración del iPhone
# Resolución original: 3024 x 4032

camera_matrix = np.array([
    [3096.95877, 0, 1468.81374],
    [0, 3106.46360, 2042.52139],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array([0.192835230,-1.17951104,-0.000110699824, -0.00323880880,2.54548115], dtype=np.float64)

CALIB_WIDTH = 3024
CALIB_HEIGHT = 4032


def main(camera_index=1, width=1280, height=720):

    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer la imagen.")
            return

        # Resolución real del vídeo
        h, w = frame.shape[:2]
        print(f"Resolución real: {w} x {h}")

        # Adaptar matriz intrínseca a la resolución
        K = camera_matrix.copy()

        K[0, 0] *= w / CALIB_WIDTH
        K[0, 2] *= w / CALIB_WIDTH

        K[1, 1] *= h / CALIB_HEIGHT
        K[1, 2] *= h / CALIB_HEIGHT

        # Nueva matriz de cámara
        new_K, roi = cv2.getOptimalNewCameraMatrix(K,dist_coeffs,(w, h),0,(w, h))

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            corrected = cv2.undistort(frame,K,dist_coeffs,None,new_K)

            # Efecto espejo
            original = cv2.flip(frame, 1)
            corrected = cv2.flip(corrected, 1)

            cv2.imshow("Camara original - iPhone", original)
            cv2.imshow("Camara corregida - iPhone", corrected)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=1, width=1280, height=720)
