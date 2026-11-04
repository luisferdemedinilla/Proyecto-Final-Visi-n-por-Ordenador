"""Vista superior del piano. Ejecutar con --help para ver las opciones."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np


NOTAS = ("Do4", "Re4", "Mi4", "Fa4", "Sol4", "La4", "Si4", "Do5")
DEDOS = {4: "pulgar", 8: "indice", 12: "corazon", 16: "anular", 20: "menique"}

# Coordenadas en pixeles de la plantilla impresa, antes de cualquier perspectiva.
IZQUIERDA, ARRIBA, DERECHA, ABAJO = 160, 160, 1120, 520
LADO = 96
MARCAS = {0: (32, 32), 1: (1152, 32), 2: (1152, 552), 3: (32, 552)}


def diccionario():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def generar_plantilla(ruta):
    papel = np.full((680, 1280), 255, dtype=np.uint8)
    for identificador, (x, y) in MARCAS.items():
        papel[y:y + LADO, x:x + LADO] = cv2.aruco.generateImageMarker(
            diccionario(), identificador, LADO
        )
    ancho = (DERECHA - IZQUIERDA) / len(NOTAS)
    for i, nota in enumerate(NOTAS):
        x = round(IZQUIERDA + i * ancho)
        fin = round(IZQUIERDA + (i + 1) * ancho)
        cv2.rectangle(papel, (x, ARRIBA), (fin, ABAJO), 0, 2)
        cv2.putText(papel, nota, (x + 8, ABAJO - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    if not cv2.imwrite(str(ruta), papel):
        raise OSError(f"No se pudo guardar {ruta}")


def localizar_teclado(detector, imagen):
    """Devuelve la homografia imagen -> papel o None si el ajuste es dudoso."""
    esquinas, ids, _ = detector.detectMarkers(imagen)
    if ids is None:
        return None

    origen, destino = [], []
    for marca, vertices in zip(ids.ravel(), esquinas):
        if marca in MARCAS:
            x, y = MARCAS[marca]
            origen.extend(vertices.reshape(4, 2))
            destino.extend([(x, y), (x + LADO, y),
                            (x + LADO, y + LADO), (x, y + LADO)])
    # Exigir al menos tres esquinas del folio para evitar extrapolar desde un lado.
    if len(origen) < 12:
        return None
    matriz, validos = cv2.findHomography(np.float32(origen), np.float32(destino),
                                         cv2.RANSAC, 5.0)
    if matriz is None or validos.sum() < 10:
        return None
    return matriz


def tecla_en(x, y):
    if IZQUIERDA <= x < DERECHA and ARRIBA <= y < ABAJO:
        return NOTAS[int((x - IZQUIERDA) * len(NOTAS) / (DERECHA - IZQUIERDA))]
    return None


class CamaraSuperior:
    """procesar recibe BGR corregido y tiempo monotono en ms; devuelve (dedos, vista).
    dedos[(mano, dedo)] contiene pixel y tecla (None fuera o sin teclado).
    Una identidad ausente significa que no hubo observacion fiable de esa mano.
    """

    def __init__(self, modelo, invertir_manos=False):
        import mediapipe as mp
        self.mp = mp
        self.invertir_manos = invertir_manos
        self.marcas = cv2.aruco.ArucoDetector(diccionario())
        vision = mp.tasks.vision
        opciones = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(modelo)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2
        )
        self.manos = vision.HandLandmarker.create_from_options(opciones)

    def procesar(self, imagen, instante_ms):
        matriz = localizar_teclado(self.marcas, imagen)
        vista = imagen.copy()
        if matriz is None:
            cv2.putText(vista, "Muestra al menos 3 marcadores", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
        else:
            limites = np.float32([[[IZQUIERDA, ARRIBA], [DERECHA, ARRIBA],
                                   [DERECHA, ABAJO], [IZQUIERDA, ABAJO]]])
            borde = cv2.perspectiveTransform(limites, np.linalg.inv(matriz))
            cv2.polylines(vista, [borde.astype(np.int32)], True, (0, 220, 0), 2)

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
            x, y, w, h = cv2.boundingRect(pixeles)
            cv2.rectangle(vista, (x, y), (x + w, y + h), (200, 160, 0), 1)
            papel = None
            if matriz is not None:
                papel = cv2.perspectiveTransform(pixeles[None], matriz)[0]
            for i, dedo in DEDOS.items():
                pixel = tuple(map(int, pixeles[i]))
                nota = tecla_en(*papel[i]) if papel is not None else None
                dedos[mano, dedo] = {"pixel": pixel, "tecla": nota}
                cv2.circle(vista, pixel, 4, (0, 220, 255), -1)
                cv2.putText(vista, f"{mano[0]}-{dedo[:3]} {nota or '-'}", pixel,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 60, 100), 1)
        return dedos, vista

    def cerrar(self):
        self.manos.close()


def demo():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plantilla", metavar="PNG", help="crear teclado para imprimir")
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--modelo", default="hand_landmarker.task")
    parser.add_argument("--calibracion", help="NPZ con camera_matrix y dist_coeffs")
    parser.add_argument("--invertir-manos", action="store_true")
    args = parser.parse_args()
    if args.plantilla:
        generar_plantilla(args.plantilla)
        return
    if not Path(args.modelo).is_file():
        parser.error(f"Falta el modelo: {args.modelo}")
    calibracion = None
    if args.calibracion:
        with np.load(args.calibracion) as datos:
            calibracion = datos["camera_matrix"], datos["dist_coeffs"]
    superior = CamaraSuperior(args.modelo, args.invertir_manos)
    camara = cv2.VideoCapture(args.camara)
    try:
        if not camara.isOpened():
            parser.error("No se pudo abrir la webcam. Prueba otro indice con --camara.")
        while True:
            ok, imagen = camara.read()
            instante = time.monotonic_ns() // 1_000_000
            if not ok:
                raise RuntimeError("La webcam ha dejado de entregar imagenes.")
            if calibracion is not None:
                imagen = cv2.undistort(imagen, *calibracion)
            _, vista = superior.procesar(imagen, instante)
            cv2.imshow("Camara superior (Q o Esc para salir)", vista)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        camara.release()
        superior.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()
