"""Prototipo: recorrer ocho teclas virtuales con un dedo.

Uso: python src/seguridad_flujo.py [--camara 1]
Con la mano fuera de la imagen, pulsa B para guardar el fondo. R reinicia y Q sale.
"""
import argparse

import cv2
import numpy as np


NOTAS = ("Do4", "Re4", "Mi4", "Fa4", "Sol4", "La4", "Si4", "Do5")
SECUENCIA = ("Do4", "Mi4", "Sol4", "Do5")
# Coordenadas relativas a la imagen: izquierda, arriba, derecha, abajo.
TECLADO = (0.10, 0.55, 0.90, 0.90)
VENTANA = "Seguridad del piano"
UMBRAL_DIFERENCIA = 30


def tecla_en(punto, limites):
    """
    Función que sirve para detectar si un dedo está 
    tocando las teclas definidas. 
    """
    x, y = punto # El punto es las coordenadas detectadas donde está el dedo
    izquierda, arriba, derecha, abajo = limites # Los limites representa el rectángulo que contiene a las teclas
    if izquierda <= x < derecha and arriba <= y < abajo: # Comprobamos si el dedo está tocando el teclado 
        return int((x - izquierda) * len(NOTAS) / (derecha - izquierda)) # Detectamos en que tecla está el dedo
    return None


def main():
    
    # Configuración de la cámara
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camara", type=int, default=1)
    args = parser.parse_args()
    camara = cv2.VideoCapture(args.camara)
    
    # Si no hemos conseguido abrir la cámara lanzamos un error  
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la camara; prueba otro indice con --camara")

    cv2.namedWindow(VENTANA) # Ventana donde se mostrará el vídeo 
    fondo = None  # Ponemos None porque todavía no tenemos una variable de referencia 
    progreso = 0
    preparada = True  # Hay que sacar el dedo del teclado entre pulsaciones.
    completada = False
    error = False
    estado = "Retira la mano y pulsa B para guardar el fondo"
    try:
        # Usamos un while True para estar constantemente capturando imágenes de la cámara 
        while True:
            ok, imagen = camara.read() 
            if not ok: # Si la imagen no se ha detectado correctamente 
                print("La camara ha dejado de entregar imagenes")
                break
            
            # Aplicamos un filtro de suavizado para reducir el ruido en la imagen
            suavizada = cv2.GaussianBlur(imagen, (7, 7), 0)
            alto, ancho = imagen.shape[:2]
            limites = (int(TECLADO[0] * ancho), int(TECLADO[1] * alto),int(TECLADO[2] * ancho), int(TECLADO[3] * alto))

            punto = None
            if fondo is not None: # Si ya se ha pulsado la tecla B 
                # Comparamos la imagen actual de la cámara con el fondo que tenemos guardado
                diferencia = cv2.absdiff(fondo, suavizada)
                # Guardamos la diferencia máxima entre los canales de color para detectar el movimiento del dedo
                diferencia = np.max(diferencia, axis=2)
                
                # Creamos una máscara binaria para diferenciar bien la región diferente del fondo
                _, mascara = cv2.threshold(diferencia, UMBRAL_DIFERENCIA, 255,cv2.THRESH_BINARY)
                kernel = np.ones ( (7, 7), np.uint8)
                
                # Buscamos el borde de los objetos blancos 
                contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if contornos:
                    # Seleccionamos el contorno que tiene más área (suponems que la mano es lo que más área tiene)
                    contorno = max(contornos, key=cv2.contourArea)
                    # Imponemos que la mano tiene que ocupar al menos un 0.3% del área total
                    if cv2.contourArea(contorno) > ancho * alto * 0.003:
                        
                        # Para detectar la punta del dedo hemos obtenido las cordenadas de el contorno de la mano y hemos buscado la mayor 
                        coordenadas = contorno.reshape(-1, 2)
                        y_max = int(coordenadas[:, 1].max())
                        
                        # Seleccionamos los putnos que están a 5 píxeles del extremos inferior
                        extremos = coordenadas[coordenadas[:, 1] >= y_max - 5]
                        
                        # Calculamos la mediana de los extremos que acabamos de calcular, de esta manera obtenemos la coordenada aproximada de x y de y 
                        punto = (int(np.median(extremos[:, 0])), y_max)
                        
                # Definimos el estado en el que estamos      
                estado = ("Dedo detectado" if punto is not None else "Muestra el dedo desde arriba")
            indice = None
            if punto is not None:
                
                # Si se ha detectado algún punto miramos que esté contenido en el teclado 
                indice = tecla_en(punto, limites)
                cv2.circle(imagen, punto, 8, (0, 0, 255), -1) # Dibujamos un círculo alrededor del dedo

            if punto is not None and indice is None:
                preparada = True
            if indice is not None and preparada and not completada:
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
                cv2.rectangle(imagen, (x0, arriba), (x1, abajo), color, 2)
                cv2.putText(imagen, nota, (x0 + 3, abajo - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(imagen, estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.putText(imagen, f"Secuencia: {progreso}/{len(SECUENCIA)}  |  B: fondo  R: reiniciar  Q: salir",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(imagen, "Orden: " + " -> ".join(SECUENCIA),
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(VENTANA, imagen)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (27, ord("q")):
                break
            if tecla == ord("b"):
                fondo = suavizada.copy()
                progreso = 0
                preparada = True
                completada = False
                error = False
                estado = "Muestra el dedo desde arriba"
            if tecla == ord("r"):
                progreso = 0
                preparada = True
                completada = False
                error = False
                estado = "Muestra el dedo desde arriba" if fondo is not None else "Retira la mano y pulsa B"
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
