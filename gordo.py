import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Clase Boid Completa ---
# (Esta clase no se modifica, es la que ya tenías)
class Boid:
    def __init__(self, x, y, ancho_mundo, alto_mundo, velocidad_inicial=None):
        # Atributos: posicion (xi) y velocidad (vi)
        self.posicion = np.array([float(x), float(y)])
        
        if velocidad_inicial is not None:
            self.velocidad = np.array(velocidad_inicial, dtype=float)
        else:
            # Velocidad inicial aleatoria ~5 m/s
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
        
        # pondero las fuerzas
        fuerza_total = (wa * vector_alineacion + 
                        ws * vector_separacion + 
                        wc * vector_cohesion)
        
        return fuerza_total

    def actualizar_estado(self, fuerza_total, T, velocidad_max=10.0):
        """
        Actualiza la velocidad y posición usando las ecuaciones del PDF.
        """
        
        # Regla para los bordes ---
        fuerza_retorno = np.zeros(2)
        margen = 3  # A qué distancia del borde empezamos a empujar
        fuerza_magnitud = 3.5 # Qué tan fuerte empujamos
        
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

        # 1. Actualizar velocidad (Ecuación 4) 
        self.velocidad = self.velocidad + fuerza_total * T 
        
        # Limitar la velocidad máxima
        norma = np.linalg.norm(self.velocidad)
        if norma > velocidad_max:
            self.velocidad = (self.velocidad / norma) * velocidad_max
        
        # 2. Actualizar posición 
        self.posicion = self.posicion + T * self.velocidad # Usamos la velocidad *nueva*

    # --- MÉTODOS DE REGLAS (sin cambios) ---
    def alineacion(self, boids, radio_a):
        vector_promedio = np.zeros(2)
        conteo = 0
        for otro in boids:
            dist = np.linalg.norm(self.posicion - otro.posicion)
            if 0 < dist < radio_a:
                vector_promedio += otro.velocidad
                conteo += 1
        if conteo > 0:
            vector_promedio /= conteo
            a_i = vector_promedio - self.velocidad
            return a_i
        else:
            return np.zeros(2)

    def separacion(self, boids, radio_s):
        vector_separacion = np.zeros(2)
        for otro in boids:
            dist = np.linalg.norm(self.posicion - otro.posicion)
            if 0 < dist < radio_s:
                diferencia = self.posicion - otro.posicion
                vector_separacion += diferencia
        return vector_separacion

    def cohesion(self, boids, radio_c):
        centro_masa = np.zeros(2)
        conteo = 0
        for otro in boids:
            dist = np.linalg.norm(self.posicion - otro.posicion)
            if 0 < dist < radio_c:
                centro_masa += otro.posicion
                conteo += 1
        if conteo > 0:
            centro_masa /= conteo
            c_i = centro_masa - self.posicion
            return c_i
        else:
            return np.zeros(2)

# --- 2. Funciones de Simulación --- (Gran aporte de gemini)

def correr_simulacion_boids(bandada, ANCHO_MUNDO, ALTO_MUNDO, T, WA, WS, WC, RA, RS, RC):
    """
    Esta función contiene AHORA toda la lógica de simulación y animación.
    Toma la 'bandada' ya creada como parámetro.
    """
    
    # --- Configurar la visualización ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ANCHO_MUNDO)
    ax.set_ylim(0, ALTO_MUNDO)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    fig.set_facecolor('black')
    ax.tick_params(colors='white')

    posiciones = np.array([b.posicion for b in bandada])
    scatter = ax.scatter(posiciones[:, 0], posiciones[:, 1], s=10, color='cyan')

    # --- Función de Animación (Anidada) ---
    # La definimos aquí para que tenga acceso a 'bandada' y los parámetros
    def actualizar_animacion(frame):
        fuerzas_calculadas = []
        
        # 1. Calcular todas las fuerzas PRIMERO
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
    print(f"Iniciando simulación de Boids con {len(bandada)} agentes...")
    ani = FuncAnimation(fig, actualizar_animacion, frames=500, interval=20, blit=True)
    plt.show()

# --- 3. Funciones de Setup (Para los casos de prueba) ---

def setup_boids_aleatorio(num_boids, ancho, alto):
    """ Caso por defecto: boids aleatorios (como lo tenías). """
    print("Configurando: Boids en posiciones aleatorias.")
    return [Boid(np.random.rand() * ancho, 
                 np.random.rand() * alto, 
                 ancho, alto) 
            for _ in range(num_boids)]

def setup_boids_caso_a(ancho, alto):
    """ Caso (a): Un boid se encuentra con otro boid. """
    print("Configurando: Caso (a) - 1 vs 1.")
    bandada = []
    # Boid 1 (izquierda, va a la derecha)
    bandada.append(Boid(ancho * 0.25, alto * 0.5, ancho, alto, velocidad_inicial=[5.0, 0.0]))
    # Boid 2 (derecha, va a la izquierda)
    bandada.append(Boid(ancho * 0.75, alto * 0.5, ancho, alto, velocidad_inicial=[-5.0, 0.0]))
    return bandada

def setup_boids_caso_d(num_por_grupo, ancho, alto):
    """ Caso (d): Dos grupos de boids del mismo tamaño se encuentran. """
    print(f"Configurando: Caso (d) - {num_por_grupo} vs {num_por_grupo}.")
    bandada = []
    # Grupo 1 (esquina inferior izquierda, va arriba-derecha)
    for _ in range(num_por_grupo):
        x = ancho * 0.2 + np.random.rand() * 2
        y = alto * 0.2 + np.random.rand() * 2
        bandada.append(Boid(x, y, ancho, alto, velocidad_inicial=[4.0, 4.0]))
    
    # Grupo 2 (esquina superior derecha, va abajo-izquierda)
    for _ in range(num_por_grupo):
        x = ancho * 0.8 + np.random.rand() * 2
        y = alto * 0.8 + np.random.rand() * 2
        bandada.append(Boid(x, y, ancho, alto, velocidad_inicial=[-4.0, -4.0]))
    return bandada

# --- 4. Interfaz Principal ---

if __name__ == "__main__":
    
    # --- Parámetros Globales de Simulación ---
    # Puedes ajustar esto para todas las simulaciones de Boids
    
    # Parámetros de Simulación (del PDF)
    T = 0.01        # Periodo de actualización
    WA = 0.1        # Peso alineación
    WS = 0.1        # Peso separación
    WC = 0.1        # Peso cohesión
    RA = 4.0        # Radio alineación 
    RS = 0.5        # Radio separación 
    RC = 4.0        # Radio cohesión 

    # Parámetros del mundo
    ANCHO_MUNDO = 15
    ALTO_MUNDO = 15
    NUM_BOIDS = 25 # Para el caso aleatorio
    NUM_GRUPO = 10 # Para el caso (d)

    # --- Bucle del Menú (La "Interfaz") ---
    while True:
        print("\n--- Interfaz de Simulaciones (TP8) ---")
        print("1. Ejecutar Actividad 2: Boids (Aleatorio)")
        print("2. Ejecutar Actividad 2: Boids (Caso a: 1 vs 1)")
        print("3. Ejecutar Actividad 2: Boids (Caso d: Grupo vs Grupo)")
        print("0. Salir")
        
        opcion = input("Seleccione una opción: ")
        
        if opcion == '1':
            bandada = setup_boids_aleatorio(NUM_BOIDS, ANCHO_MUNDO, ALTO_MUNDO)
            correr_simulacion_boids(bandada, ANCHO_MUNDO, ALTO_MUNDO, T, WA, WS, WC, RA, RS, RC)
            
        elif opcion == '2':
            bandada = setup_boids_caso_a(ANCHO_MUNDO, ALTO_MUNDO)
            correr_simulacion_boids(bandada, ANCHO_MUNDO, ALTO_MUNDO, T, WA, WS, WC, RA, RS, RC)
            
        elif opcion == '3':
            bandada = setup_boids_caso_d(NUM_GRUPO, ANCHO_MUNDO, ALTO_MUNDO)
            correr_simulacion_boids(bandada, ANCHO_MUNDO, ALTO_MUNDO, T, WA, WS, WC, RA, RS, RC)

        elif opcion == '0':
            print("Saliendo.")
            break
            
        else:
            print("Opción no válida. Intente de nuevo.")