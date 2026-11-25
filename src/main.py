from camara_frontal import *
from camara_superior import *
from piano import *

def main():
    camara_frontal = CamaraFrontal("src/hand_landmarker.task")
    camara_superior = CamaraSuperior("src/hand_landmarker.task")
    piano = Piano()

    # Cámaras frontal y superior respectivamente
    camara = cv2.VideoCapture(0)
    camara_2 = cv2.VideoCapture(1)

    while True:
        ret, frame = camara.read()
        ret2, frame2 = camara_2.read()

        if not ret or not ret2:
            print("Error al capturar la imagen de la cámara.")
            break

        instante = time.monotonic_ns() // 1_000_000

        dedos_frontal, vista_frontal = camara_frontal.procesar(frame, instante)
        dedos_superior, vista_superior = camara_superior.procesar(frame2, instante)

        cv2.imshow("Camara Frontal", vista_frontal)
        cv2.imshow("Camara Superior", vista_superior)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

        for (mano, dedo), info in dedos_frontal.items():
            tecla = info["toca"]

            if tecla:
                piano.pulsar(mano, dedo, tecla)
            else:
                piano.soltar(mano, dedo)

if __name__ == "__main__":
    main()
                
            
