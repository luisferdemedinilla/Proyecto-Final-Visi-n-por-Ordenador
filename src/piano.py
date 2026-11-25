from pathlib import Path
import pygame


class Piano:
    def __init__(self):
        pygame.mixer.init(buffer=512)
        pygame.mixer.set_num_channels(32)

        carpeta ="sounds/"
        notas = ["Do4", "Re4", "Mi4", "Fa4",
                 "Sol4", "La4", "Si4", "Do5"]

        self.sonidos = {
            nota: pygame.mixer.Sound(carpeta + f"{nota}.wav")
            for nota in notas
        }

        self.activos = {}

    def pulsar(self, mano, dedo, nota):
        identidad = (mano, dedo)

        # Mantener el dedo apoyado no vuelve a iniciar el sonido.
        if identidad in self.activos:
            return

        canal = self.sonidos[nota].play()

        if canal is not None:
            self.activos[identidad] = canal

    def soltar(self, mano, dedo):
        canal = self.activos.pop((mano, dedo), None)

        if canal is not None:
            canal.fadeout(120)

    def cerrar(self):
        pygame.mixer.quit()
    