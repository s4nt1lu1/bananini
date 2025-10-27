import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Clase Boid Completa ---

class Boid:
    def __init__(self, x, y, ancho_mundo, alto_mundo):
        # Atributos: posicion (xi) y velocidad (vi) [cite: 42]
        self.posicion = np.array([float(x), float(y)])
        
        # Velocidad inicial aleatoria ~5 m/s [cite: 92, 93]
        angulo = np.random.rand() * 2 * np.pi
        self.velocidad = np.array([np.cos(angulo), np.sin(angulo)]) * 5.0
        
        self.ancho_mundo = ancho_mundo
        self.alto_mundo = alto_mundo

    def calcular_fuerzas(self, boids, wa, ws, wc, ra, rs, rc):
        """
        Calcula las 3 fuerzas (vectores de dirección) de las reglas.
        """
        vector_alineacion = self.alineacion(boids, ra)
        vector_separacion = self.separacion(boids, rs)
        vector_cohesion = self.cohesion(boids, rc)
        
        # Ponderar las fuerzas como en la Ecuación 4 [cite: 70]
        fuerza_total = (wa * vector_alineacion + 
                        ws * vector_separacion + 
                        wc * vector_cohesion)
        
        return fuerza_total

    def actualizar_estado(self, fuerza_total, T, velocidad_max=10.0):
        """
        Actualiza la velocidad y posición usando las ecuaciones del PDF.
        """
        
        # --- Implementación de la Regla de Borde [cite: 80-83] ---
        fuerza_retorno = np.zeros(2)
        margen = 20  # A qué distancia del borde empezamos a empujar
        fuerza_magnitud = 0.5 # Qué tan fuerte empujamos
        
        if self.posicion[0] < margen:
            fuerza_retorno[0] = (margen - self.posicion[0]) * fuerza_magnitud
        elif self.posicion[0] > self.ancho_mundo - margen:
            fuerza_retorno[0] = (self.ancho_mundo - margen - self.posicion[0]) * fuerza_magnitud
            
        if self.posicion[1] < margen:
            fuerza_retorno[1] = (margen - self.posicion[1]) * fuerza_magnitud
        elif self.posicion[1] > self.alto_mundo - margen:
            fuerza_retorno[1] = (self.alto_mundo - margen - self.posicion[1]) * fuerza_magnitud
        
        fuerza_total += fuerza_retorno
        # --- Fin de la Regla de Borde ---

        # 1. Actualizar velocidad (Ecuación 4) [cite: 70]
        self.velocidad = self.velocidad + fuerza_total * T # Multiplicamos por T (aunque en Ec. 4 no está, es físicamente más correcto)
        
        # Limitar la velocidad máxima
        norma = np.linalg.norm(self.velocidad)
        if norma > velocidad_max:
            self.velocidad = (self.velocidad / norma) * velocidad_max
        
        # 2. Actualizar posición [cite: 72]
        self.posicion = self.posicion + T * self.velocidad # Usamos la velocidad *nueva*

    # --- MÉTODOS DE REGLAS COMPLETADOS ---

    def alineacion(self, boids, radio_a):
        """
        Regla de Alineación (Ecuación 1) [cite: 48, 53]
        """
        vector_promedio = np.zeros(2)
        conteo = 0
        
        for otro in boids:
            dist = np.linalg.norm(self.posicion - otro.posicion)
            # Vecindad de alineación Ai [cite: 65]
            if 0 < dist < radio_a:
                vector_promedio += otro.velocidad
                conteo += 1
                
        # Salvar indeterminación si la vecindad está vacía [cite: 67]
        if conteo > 0:
            vector_promedio /= conteo
            a_i = vector_promedio - self.velocidad
            return a_i
        else:
            return np.zeros(2) # Influencia nula [cite: 68]

    def separacion(self, boids, radio_s):
        """
        Regla de Separación (Ecuación 2) [cite: 55, 57]
        """
        vector_separacion = np.zeros(2)
        
        for otro in boids:
            dist = np.linalg.norm(self.posicion - otro.posicion)
            # Vecindad de separación Si
            if 0 < dist < radio_s:
                diferencia = self.posicion - otro.posicion
                # La ecuación es suma(xi - xj)
                vector_separacion += diferencia
                
        return vector_separacion

    def cohesion(self, boids, radio_c):
        """
        Regla de Cohesión (Ecuación 3) [cite: 59, 61]
        """
        centro_masa = np.zeros(2)
        conteo = 0
        
        for otro in boids:
            dist = np.linalg.norm(self.posicion - otro.posicion)
            # Vecindad de cohesión Ci [cite: 62]
            if 0 < dist < radio_c:
                centro_masa += otro.posicion
                conteo += 1
                
        # Salvar indeterminación si la vecindad está vacía [cite: 67]
        if conteo > 0:
            centro_masa /= conteo
            c_i = centro_masa - self.posicion
            return c_i
        else:
            return np.zeros(2) # Influencia nula [cite: 68]

# --- 2. La Simulación y Animación ---

# Parámetros de Simulación (del PDF) [cite: 75, 76]
T = 0.01        # Periodo (aumentado para ver mejor la simulación)
WA = 0.1        # Peso alineación
WS = 0.1        # Peso separación
WC = 0.1        # Peso cohesión
RA = 10.0       # Radio alineación (aumentado para un mundo más grande)
RS = 1.0        # Radio separación (aumentado para un mundo más grande)
RC = 10.0       # Radio cohesión (aumentado para un mundo más grande)

# Parámetros del mundo
ANCHO_MUNDO = 100
ALTO_MUNDO = 100
NUM_BOIDS = 50

# --- Crear la bandada ---
bandada = [Boid(np.random.rand() * ANCHO_MUNDO, 
                np.random.rand() * ALTO_MUNDO, 
                ANCHO_MUNDO, ALTO_MUNDO) 
           for _ in range(NUM_BOIDS)]

# --- Configurar la visualización ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, ANCHO_MUNDO)
ax.set_ylim(0, ALTO_MUNDO)
ax.set_aspect('equal')
ax.set_facecolor('black')
fig.set_facecolor('black')
ax.tick_params(colors='white')

# Usaremos scatter para los boids.
posiciones = np.array([b.posicion for b in bandada])
scatter = ax.scatter(posiciones[:, 0], posiciones[:, 1], s=10, color='cyan')

# --- Función de Animación (El "paso temporal") ---
def actualizar_animacion(frame):
    
    fuerzas_calculadas = []
    
    # 1. Calcular todas las fuerzas PRIMERO (basado en estado t-T)
    for boid in bandada:
        fuerza = boid.calcular_fuerzas(bandada, WA, WS, WC, RA, RS, RC)
        fuerzas_calculadas.append(fuerza)
        
    # 2. Actualizar todos los estados DESPUÉS
    posiciones_nuevas = []
    for i, boid in enumerate(bandada):
        boid.actualizar_estado(fuerzas_calculadas[i], T)
        posiciones_nuevas.append(boid.posicion)
        
    # 3. Actualizar el dibujo
    scatter.set_offsets(posiciones_nuevas)
    
    return scatter,

# --- Correr la Animación ---
ani = FuncAnimation(fig, actualizar_animacion, frames=500, interval=20, blit=True)

plt.show()