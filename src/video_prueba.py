
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camara encontrada en indice {i}")
            cv2.imshow(f"Camara {i}", frame)
            cv2.waitKey(2000)

    cap.release()
    cv2.destroyAllWindows()
