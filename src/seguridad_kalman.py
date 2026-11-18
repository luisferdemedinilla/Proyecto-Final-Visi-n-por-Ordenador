"""Secuencia de teclas mediante MeanShift y filtro de Kalman (apartado D).

Coloca una marca de color vivo en la punta del dedo. Pulsa S y selecciona
la marca con el raton; confirma la seleccion con Espacio o Enter.
"""
import argparse

import cv2
import numpy as np

from seguridad_flujo import NOTAS, SECUENCIA, TECLADO, tecla_en


VENTANA = "Seguridad - MeanShift y Kalman"
CRITERIO = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)


def crear_filtro(x, y):
    filtro = cv2.KalmanFilter(4, 2)
    filtro.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    filtro.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                        [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    filtro.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    filtro.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2
    filtro.errorCovPost = np.eye(4, dtype=np.float32)
    filtro.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return filtro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camara", type=int, default=1)
    args = parser.parse_args()
    camara = cv2.VideoCapture(args.camara)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la camara")

    histograma = ventana = filtro = None
    progreso = 0
    preparada = True
    completada = False
    error = False
    candidato = None
    consecutivos = 0
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, imagen = camara.read()
            if not ok:
                print("La camara ha dejado de entregar imagenes")
                break
            alto, ancho = imagen.shape[:2]
            limites = (int(TECLADO[0] * ancho), int(TECLADO[1] * alto),
                       int(TECLADO[2] * ancho), int(TECLADO[3] * alto))
            vista = imagen.copy()
            punto = None
            estado = "Pulsa S y selecciona la marca de color"
            if histograma is not None:
                hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
                proyeccion = cv2.calcBackProject([hsv], [0], histograma, [0, 180], 1)
                saturados = cv2.inRange(hsv, (0, 70, 50), (180, 255, 255))
                proyeccion = cv2.bitwise_and(proyeccion, saturados)
                _, ventana = cv2.meanShift(proyeccion, ventana, CRITERIO)
                x, y, w, h = ventana
                # No aceptar una prediccion si no hay suficientes pixeles observados.
                region = proyeccion[y:y + h, x:x + w]
                if region.size and np.mean(region) >= 30:
                    cx, cy = x + w / 2, y + h / 2
                    filtro.predict()
                    corregido = filtro.correct(np.array([[cx], [cy]], np.float32))
                    punto = (int(corregido[0, 0]), int(corregido[1, 0]))
                    cv2.rectangle(vista, (x, y), (x + w, y + h), (255, 180, 0), 2)
                    cv2.circle(vista, punto, 7, (0, 0, 255), -1)
                    estado = "Marca detectada"
                else:
                    histograma = ventana = filtro = None
                    preparada = False
                    progreso = 0
                    error = False
                    consecutivos = 0
                    candidato = None
                    estado = "Marca perdida: pulsa S para seleccionarla de nuevo"

            indice = tecla_en(punto, limites) if punto is not None else None
            # Tres imagenes consecutivas reducen los cambios por ruido en los bordes.
            if punto is not None:
                if indice == candidato:
                    consecutivos += 1
                else:
                    candidato, consecutivos = indice, 1
                if consecutivos >= 3:
                    if indice is None:
                        preparada = True
                    elif preparada and not completada:
                        preparada = False
                        if NOTAS[indice] == SECUENCIA[progreso]:
                            progreso += 1
                            error = False
                            completada = progreso == len(SECUENCIA)
                        else:
                            progreso = 0
                            error = True
            if completada:
                estado = "Secuencia correcta - R para reiniciar"
            elif error:
                estado = "Tecla incorrecta: vuelve a empezar"

            izquierda, arriba, derecha, abajo = limites
            for i, nota in enumerate(NOTAS):
                x0 = izquierda + i * (derecha - izquierda) // len(NOTAS)
                x1 = izquierda + (i + 1) * (derecha - izquierda) // len(NOTAS)
                color = (0, 220, 0) if i == indice else (255, 255, 255)
                cv2.rectangle(vista, (x0, arriba), (x1, abajo), color, 2)
                cv2.putText(vista, nota, (x0 + 3, abajo - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for linea, texto in enumerate((estado,
                    f"Secuencia: {progreso}/{len(SECUENCIA)}  |  S: seleccionar  R: reiniciar  Q: salir",
                    "Orden: " + " -> ".join(SECUENCIA))):
                cv2.putText(vista, texto, (10, 30 + linea * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(VENTANA, vista)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord("r"):
                progreso, preparada, completada, error = 0, False, False, False
                candidato, consecutivos = None, 0
            if tecla == ord("s"):
                # Seleccion sobre el fotograma original, sin las teclas dibujadas.
                x, y, w, h = cv2.selectROI(VENTANA, imagen, False, False)
                if w and h:
                    hsv = cv2.cvtColor(imagen[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
                    mascara = cv2.inRange(hsv, (0, 70, 50), (180, 255, 255))
                    if cv2.countNonZero(mascara) < w * h * 0.15:
                        print("Selecciona una marca de color vivo, con poca piel o fondo")
                        continue
                    histograma = cv2.calcHist([hsv], [0], mascara, [180], [0, 180])
                    cv2.normalize(histograma, histograma, 0, 255, cv2.NORM_MINMAX)
                    ventana = (x, y, w, h)
                    filtro = crear_filtro(x + w / 2, y + h / 2)
                    progreso, preparada, completada, error = 0, False, False, False
                    candidato, consecutivos = None, 0
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
