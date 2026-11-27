import time

import cv2
import pygame

from camara_frontal import CamaraFrontal
from camara_superior import CamaraSuperior
from interfaz import InterfazPiano
from piano import Piano
from seguridad_kalman import solicitar_acceso


def tocar(piano):
    modelo = "src/hand_landmarker.task"
    camara = cv2.VideoCapture(0)
    camara_2 = camara_frontal = camara_superior = interfaz = None
    notas_pulsadas = {}
    try:
        camara_2 = cv2.VideoCapture(1)
        for captura, indice, ancho, alto in (
            (camara, 0, 1920, 1080),
            (camara_2, 1, 1280, 720),
        ):
            if not captura.isOpened():
                raise RuntimeError(f"No se pudo abrir la cámara {indice}")
            captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)

        ok, frame = camara.read()
        if not ok:
            raise RuntimeError("No se pudo capturar la imagen frontal para seleccionar las marcas")

        camara_frontal = CamaraFrontal(modelo, imagen=frame)
        if camara_frontal.marcas is None:
            return False
        camara_superior = CamaraSuperior(modelo)
        interfaz = InterfazPiano()
        reloj = pygame.time.Clock()

        while True:
            accion = interfaz.accion()
            if accion is not None:
                return accion == "reiniciar"

            ok, frame = camara.read()
            ok_superior, frame_2 = camara_2.read()
            if not ok or not ok_superior:
                raise RuntimeError("Una cámara ha dejado de entregar imágenes")

            instante = time.monotonic_ns() // 1_000_000
            dedos_frontal, vista_frontal = camara_frontal.procesar(frame, instante)
            dedos_superior, vista_superior = camara_superior.procesar(frame_2, instante)

            pulsaciones = {}
            for identidad, datos in dedos_frontal.items():
                superior = dedos_superior.get(identidad)
                if datos["toca"] and superior is not None and superior["tecla"] is not None:
                    pulsaciones[identidad] = superior["tecla"]

            for identidad in notas_pulsadas.keys() - pulsaciones.keys():
                piano.soltar(*identidad)
                del notas_pulsadas[identidad]

            for identidad, nota in pulsaciones.items():
                if identidad not in notas_pulsadas:
                    piano.pulsar(*identidad, nota)
                    if identidad in piano.activos:
                        notas_pulsadas[identidad] = nota

            reloj.tick()
            interfaz.mostrar(vista_superior, vista_frontal,
                             set(notas_pulsadas.values()), reloj.get_fps())
    finally:
        for identidad in notas_pulsadas:
            piano.soltar(*identidad)
        camara.release()
        if camara_2 is not None:
            camara_2.release()
        if camara_frontal is not None:
            camara_frontal.cerrar()
        if camara_superior is not None:
            camara_superior.cerrar()
        if interfaz is not None:
            interfaz.cerrar()
        cv2.destroyAllWindows()


def main():
    piano = Piano()
    try:
        while solicitar_acceso(piano, indice=1):
            if not tocar(piano):
                break
    finally:
        piano.cerrar()


if __name__ == "__main__":
    main()
