import cv2
import pygame

from camara_superior import NOTAS


class InterfazPiano:
    def __init__(self):
        pygame.font.init()
        self.pantalla = pygame.display.set_mode((900, 565))
        pygame.display.set_caption("Virtual Piano")
        self.titulo = pygame.font.SysFont("segoeui", 24, bold=True)
        self.normal = pygame.font.SysFont("segoeui", 16)
        self.negrita = pygame.font.SysFont("segoeui", 16, bold=True)
        self.reinicio = pygame.Rect(16, 497, 868, 50)

    def accion(self):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return "salir"
            if evento.type == pygame.KEYDOWN and evento.key in (pygame.K_ESCAPE, pygame.K_q):
                return "salir"
            if (evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1
                    and self.reinicio.collidepoint(evento.pos)):
                return "reiniciar"
        return None

    def texto(self, texto, posicion, fuente=None, color="#202422", centrado=False):
        fuente = fuente or self.normal
        imagen = fuente.render(texto, True, color)
        rectangulo = imagen.get_rect(center=posicion) if centrado else imagen.get_rect(topleft=posicion)
        self.pantalla.blit(imagen, rectangulo)

    def camara(self, vista, x, rotulo):
        marco = pygame.Rect(x, 65, 428, 241)
        alto, ancho = vista.shape[:2]
        escala = min(marco.width / ancho, marco.height / alto)
        tamano = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
        rgb = cv2.cvtColor(cv2.resize(vista, tamano), cv2.COLOR_BGR2RGB)
        imagen = pygame.image.frombuffer(rgb, tamano, "RGB")
        panel = pygame.Surface(marco.size, pygame.SRCALPHA)
        panel.fill("#242826")
        panel.blit(imagen, imagen.get_rect(center=panel.get_rect().center))
        mascara = pygame.Surface(marco.size, pygame.SRCALPHA)
        pygame.draw.rect(mascara, "white", mascara.get_rect(), border_radius=18)
        panel.blit(mascara, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.pantalla.blit(panel, marco)
        self.texto(rotulo, (x, 313), color="#565c58")

    def mostrar(self, superior, frontal, activas, fps):
        self.pantalla.fill("#fdfdfd")
        self.texto("♫  Virtual Piano", (16, 15), self.titulo)
        pygame.draw.rect(self.pantalla, "#d9f5e5", (724, 17, 160, 28), border_radius=14)
        self.texto("Acceso autorizado", (804, 31), self.negrita, "#167247", centrado=True)

        self.camara(superior, 16, "Cámara superior · Detección de teclas")
        self.camara(frontal, 456, "Cámara frontal · Detección de pulsaciones")

        for indice, nota in enumerate(NOTAS):
            tecla = pygame.Rect(16 + indice * 109, 348, 105, 94)
            fondo = "#b5f2d0" if nota in activas else "#ffffff"
            pygame.draw.rect(self.pantalla, fondo, tecla, border_radius=7)
            pygame.draw.rect(self.pantalla, "#a0aaa5", tecla, width=1, border_radius=7)
            self.texto(nota[:-1], (tecla.centerx, tecla.bottom - 18),
                       self.negrita, centrado=True)

        notas = ", ".join(nota[:-1] for nota in NOTAS if nota in activas)
        self.texto(f"Notas activas: {notas or '—'}", (16, 459), color="#565c58")
        estado_fps = f"FPS: {fps:.0f}"
        self.texto(estado_fps, (884 - self.normal.size(estado_fps)[0], 459), color="#565c58")

        fondo = "#e9eeeb" if self.reinicio.collidepoint(pygame.mouse.get_pos()) else "#f3f5f4"
        pygame.draw.rect(self.pantalla, fondo, self.reinicio, border_radius=8)
        self.texto("Reiniciar sesión", self.reinicio.center, self.negrita, centrado=True)
        pygame.display.flip()

    def cerrar(self):
        pygame.display.quit()
