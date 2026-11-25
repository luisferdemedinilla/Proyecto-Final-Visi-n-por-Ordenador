
import cv2
import numpy as np

# Calibración de la cámara del ordenador (1920x1080)
camera_matrix = np.array([
    [1231.43596, 0, 842.65881],
    [0, 1233.59935, 549.94979],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array([-0.00151558,0.1606347,0.00724596,-0.04527113,-0.18764772], dtype=np.float64)


def main(camera_index=0):

    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()

    if not ret:
        cap.release()
        return

    h, w = frame.shape[:2]
    print("Resolución real:", w, h)

    # Adaptar la matriz a la resolución real
    K = camera_matrix.copy()

    K[0, 0] *= w / 1920
    K[0, 2] *= w / 1920

    K[1, 1] *= h / 1080
    K[1, 2] *= h / 1080

    new_K, _ = cv2.getOptimalNewCameraMatrix(K,dist_coeffs,(w, h),0,(w, h))

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        corrected = cv2.undistort(frame,K,dist_coeffs,None,new_K)

        # Efecto espejo
        original = cv2.flip(frame, 1)
        corrected = cv2.flip(corrected, 1)

        # Reducir visualización
        original = cv2.resize(original, (960, 540))
        corrected = cv2.resize(corrected, (960, 540))

        cv2.imshow("Camara original", original)
        cv2.imshow("Camara corregida", corrected)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(camera_index=0)
