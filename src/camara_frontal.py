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


    def __init__(self, modelo, invertir_manos=False,imagen = None, tipo_detectcion = "S"):
        import mediapipe as mp
        self.threashold = 5
        self.mp = mp
        self.invertir_manos = invertir_manos
        self.marcas = calcular_marcas(imagen)
        self.tipo_detectcion = tipo_detectcion
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
        if self.tipo_detectcion == "S":
            ymax = max(self.marcas[0][1], self.marcas[1][1])
        elif self.tipo_detectcion == "I":
            ymax = min(self.marcas[0][1], self.marcas[1][1])
        else:
            ymax = (self.marcas[0][1] + self.marcas[1][1]) / 2
        if  ymax< y < ymax + self.threashold:
            return True
        return False

    def procesar(self, imagen, instante_ms):
        
        vista = imagen.copy()
      

        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        entrada = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        resultado = self.manos.detect_for_video(entrada, instante_ms)
        alto, ancho = imagen.shape[:2]
        etiquetas = [categorias[0].category_name for categorias in resultado.handedness]
        nombres = {"Left": "izquierda", "Right": "derecha"}
        if self.invertir_manos:
            nombres = {"Left": "derecha", "Right": "izquierda"}

        dedos = {}
        for etiqueta, puntos in zip(etiquetas, resultado.hand_landmarks):
            if etiquetas.count(etiqueta) != 1 or etiqueta not in nombres:
                continue
            mano = nombres[etiqueta]
            pixeles = np.float32([(p.x * ancho, p.y * alto) for p in puntos])
            for i, dedo in DEDOS.items():
                pixel = tuple(map(int, pixeles[i]))
  
                dedos[mano, dedo] = {"pixel": pixel, "toca": self.toca_tecla(pixel)}
               
        return dedos, vista

    def cerrar(self):
        self.manos.close()