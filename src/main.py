from camara_frontal import CamaraFrontal
from camara_superior import CamaraSuperior
from piano import Piano

from pathlib import Path
import cv2
import time


def main():

    modelo = "src/hand_landmarker.task"

    # Abrir las cámaras frontal y superior
    camara = cv2.VideoCapture(0)
    camara_2 = cv2.VideoCapture(1)

    # Configurar resolución de la cámara frontal
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Configurar resolución de la cámara superior
    camara_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camara_frontal = None
    camara_superior = None

    try:
        # Capturar imagen inicial para seleccionar las marcas
        ret, frame = camara.read()

        if not ret:
            print("Error al capturar la imagen frontal.")
            return

        # Inicializar los objetos
        camara_frontal = CamaraFrontal(
            modelo, imagen=frame
        )

        camara_superior = CamaraSuperior(modelo)

        piano = Piano()

        while True:

            ret, frame = camara.read()
            ret2, frame2 = camara_2.read()

            if not ret or not ret2:
                print("Error al capturar la imagen de la cámara.")
                break

            instante = time.monotonic_ns() // 1_000_000

            dedos_frontal, vista_frontal = camara_frontal.procesar(
                frame, instante
            )

            dedos_superior, vista_superior = camara_superior.procesar(
                frame2, instante
            )

            cv2.imshow("Camara Frontal", vista_frontal)
            cv2.imshow("Camara Superior", vista_superior)

            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            # Combinar la información de ambas cámaras
            # Ejecutar en cada iteración, después de procesar ambas imágenes.
            for identidad, datos in dedos_frontal.items():
                mano, dedo = identidad
                superior = dedos_superior.get(identidad)

                if datos["toca"]:
                    if superior is not None and superior["tecla"] is not None:
                        piano.pulsar(mano, dedo, superior["tecla"])
                else:
                    piano.soltar(mano, dedo)

    finally:

        camara.release()
        camara_2.release()

        if camara_frontal is not None:
            camara_frontal.cerrar()

        if camara_superior is not None:
            camara_superior.cerrar()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()