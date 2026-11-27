import argparse
from pathlib import Path
import time

import cv2
import numpy as np
DEDOS = {4: "pulgar", 8: "indice", 12: "corazon", 16: "anular", 20: "menique"}

def calcular_marcas(imagen):
    puntos = []

    def seleccionar(evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
            puntos.append((x, y))

    cv2.namedWindow("Marcas")
    cv2.setMouseCallback("Marcas", seleccionar)

    while True:
        vista = imagen.copy()

        for punto in puntos:
            cv2.circle(vista, punto, 5, (0, 0, 255), -1)

        if len(puntos) == 2:
            cv2.line(vista, puntos[0], puntos[1], (0, 255, 0), 2)

        cv2.imshow("Marcas", vista)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and len(puntos) == 2:  # Enter para confirmar
            break
        if key == ord("r"):                # R para repetir
            puntos.clear()
        if key == 27:                     # Esc para cancelar
            cv2.destroyWindow("Marcas")
            return None, None

    cv2.destroyWindow("Marcas")
    return puntos[0], puntos[1]

class CamaraFrontal:
    """procesar recibe BGR corregido y tiempo monotono en ms; devuelve (dedos, vista).
    dedos[(mano, dedo)] contiene pixel y tecla (None fuera o sin teclado).
    Una identidad ausente significa que no hubo observacion fiable de esa mano.
    """


    def __init__(self, modelo, invertir_manos=False,imagen = None):
        import mediapipe as mp
        self.threashold = 100
        self.mp = mp
        self.invertir_manos = invertir_manos
        p1, p2 = calcular_marcas(imagen)

        margen = 10

        if p1 is not None and p2 is not None:
            self.marcas = (
                (min(p1[0], p2[0]), min(p1[1], p2[1]) - margen),
                (max(p1[0], p2[0]), max(p1[1], p2[1]) + margen)
            )
        else:
            self.marcas = None
        
        vision = mp.tasks.vision
        opciones = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(modelo)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2
        )
        self.manos = vision.HandLandmarker.create_from_options(opciones)

    def toca_tecla(self, pixel):
        if self.marcas is None:
            return False
        x, y = pixel
        if self.marcas[0][1] < y < self.marcas[1][1] or self.marcas[1][1] < y < self.marcas[0][1]:  
            return True
        return False

    def procesar(self, imagen, instante_ms):

        # Corregir la distorsión de la cámara

        vista = imagen.copy()

        # Dibujar la línea de referencia
        # Dibujar la zona de contacto con relleno transparente
        if self.marcas is not None and None not in self.marcas:
            capa = vista.copy()

            cv2.rectangle(
                capa,
                self.marcas[0],
                self.marcas[1],
                (0, 255, 0),
                -1
            )

            # 0.2 representa un 20 % de opacidad
            cv2.addWeighted(capa, 0.2, vista, 0.8, 0, dst=vista)

            # Dibujar el borde
            cv2.rectangle(
                vista,
                self.marcas[0],
                self.marcas[1],
                (0, 255, 0),
                2
            )
        # Detección de manos
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        entrada = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb
        )

        resultado = self.manos.detect_for_video(entrada, instante_ms)

        alto, ancho = imagen.shape[:2]

        etiquetas = [
            categorias[0].category_name
            for categorias in resultado.handedness
        ]

        nombres = {"Left": "izquierda", "Right": "derecha"}

        if self.invertir_manos:
            nombres = {"Left": "derecha", "Right": "izquierda"}

        dedos = {}

        for etiqueta, puntos in zip(etiquetas, resultado.hand_landmarks):

            if etiquetas.count(etiqueta) != 1 or etiqueta not in nombres:
                continue

            mano = nombres[etiqueta]

            pixeles = np.float32([
                (p.x * ancho, p.y * alto) for p in puntos
            ])

            # Dibujar rectángulo alrededor de la mano
            x, y, w, h = cv2.boundingRect(pixeles)

            cv2.rectangle(
                vista, (x, y), (x + w, y + h),
                (200, 160, 0), 1
            )

            for i, dedo in DEDOS.items():

                pixel = tuple(map(int, pixeles[i]))
                toca = self.toca_tecla(pixel)

                dedos[mano, dedo] = {
                    "pixel": pixel,
                    "toca": toca
                }

                # Verde si toca, rojo si no toca
                color = (0, 255, 0) if toca else (0, 0, 255)

                cv2.circle(vista, pixel, 5, color, -1)

                cv2.putText(
                    vista,
                    f"{mano[0]}-{dedo[:3]} {toca}",
                    pixel,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1
                )

        return dedos, vista

    def cerrar(self):
        self.manos.close()