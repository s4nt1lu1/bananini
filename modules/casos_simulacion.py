import numpy as np
class Boid():
    def __init__(self, x_i, v_i, bordes, margenes):
        self.x = np.array(x_i)
        self.v = np.array(v_i)

        self.v_min = 5 # m/s
        self.v_max = 15

        self.borderX_min, self.borderX_max = bordes[0]
        self.borderY_min, self.borderY_max = bordes[1]

        self.worldX_min, self.worldX_max = margenes[0]
        self.worldY_min, self.worldY_max = margenes[1]

        self.wa = 0.1
        self.ws = 0.1
        self.wc = 0.1
        self.wb = 2 # peso repulsión del borde
        self.wr = 0.05 # peso ruido aleatorio
        self.rhoa = 4 # metros
        self.rhos = 1.5
        self.rhoc = 4

        self.T = 0.001 # segundos
    
    def actualizar(self, boids):
        # Alineación
        boids = np.array(boids)
        diferencias_alineacion = np.zeros(2)
        conteo_alineacion = 0
        for i in range(len(boids)):
            norma = np.linalg.norm(boids[i].x-self.x)
            if norma <= self.rhoa:
                diferencias_alineacion += boids[i].v-self.v
                conteo_alineacion += 1

        if conteo_alineacion > 0:
            diferencias_alineacion /= conteo_alineacion 
        
        # Separación
        diferencias_separacion = np.zeros(2)
        for i in range(len(boids)):
            # Esta fuerza, por cómo se plantea en la actividad,
            # ve reducida su magnitud mientras más se acercan los boids.
            # Aquí proponemos esta alternativa.
            norma = np.linalg.norm(self.x-boids[i].x) 
            if 0 < norma <= self.rhos:
                diferencias_separacion += (self.x - boids[i].x)/norma

        # Cohesión
        diferencias_cohesion = np.zeros(2)
        conteo_cohesion = 0
        for i in range(len(boids)):
            diff_cohesion = boids[i].x-self.x
            norma = np.linalg.norm(diff_cohesion)
            if norma <= self.rhoc:
                diferencias_separacion += diff_cohesion
                conteo_cohesion += 1
        
        if conteo_cohesion > 0:
            diferencias_cohesion /= conteo_cohesion

        # MÁRGENES
        fuerza_retorno = np.zeros(2)

        # --- Eje X ---
        if self.x[0] < self.borderX_min:
            # empujar hacia la derecha
            fuerza_retorno[0] = (self.borderX_min - self.x[0])

        elif self.x[0] > self.borderX_max:
            # empujar hacia la izquierda
            fuerza_retorno[0] = (self.borderX_max - self.x[0])

        # --- Eje Y ---
        if self.x[1] < self.borderY_min:
            # empujar hacia arriba (en tu sistema físico)
            fuerza_retorno[1] = (self.borderY_min - self.x[1])

        elif self.x[1] > self.borderY_max:
            # empujar hacia abajo
            fuerza_retorno[1] = (self.borderY_max - self.x[1])


        ai = diferencias_alineacion
        si = diferencias_separacion
        ci = diferencias_cohesion

        # --- Ruido lateral (desviación izquierda/derecha) ---
        if np.linalg.norm(self.v) > 0:
            dir_normal = np.array([-self.v[1] + 0., self.v[0]])  
            dir_normal /= np.linalg.norm(dir_normal)        
            ruido = dir_normal * (np.random.rand() * 2 - 1) 
        else:
            ruido = np.zeros(2)

        self.v = self.v + self.wa * ai + self.ws * si + self.wc * ci + self.wb * fuerza_retorno + self.wr * ruido
        v_actual = np.linalg.norm(self.v)
        if v_actual >= self.v_max:
            self.v = (self.v/v_actual) * self.v_max
        
        if v_actual <= self.v_min:
            self.v = (self.v/v_actual) * self.v_min

        self.x = self.x + self.T * self.v


def crear_caso(opcion, bordes, margenes):
    boids = []
    if opcion == 1:
        # (a) Dos boids se encuentran de frente
        boids = [
            Boid(
                [bordes[0][0], (margenes[1][1] + margenes[1][0]) / 2],
                [5, 0], bordes, margenes
            ),
            Boid(
                [bordes[0][1], (margenes[1][1] + margenes[1][0]) / 2],
                [-5, 0], bordes, margenes
            )
        ]

    elif opcion == 2:
        # (b) Varios boids se encuentran
        # Boids distribuidos aleatoriamente en el mundo
        num_boids = 12
        boids = []
        for _ in range(num_boids):
            x0 = np.random.uniform(bordes[0][0], bordes[0][1])
            y0 = np.random.uniform(bordes[1][0], bordes[1][1])

            # dirección aleatoria con velocidad moderada
            ang = np.random.uniform(0, 2 * np.pi)
            v0 = np.array([np.cos(ang), np.sin(ang)]) * np.random.uniform(5, 10)

            boids.append(Boid([x0, y0], v0, bordes, margenes))

    elif opcion == 3:
        # (c) Un boid se encuentra con un grupo de boids
        y_centro = (margenes[1][1] + margenes[1][0]) / 2
        boids = [
            Boid(
                [bordes[0][0] + 1, y_centro],
                [6, 0], bordes, margenes
            )
        ]
        for i in range(4):
            boids.append(
                Boid(
                    [bordes[0][1] - 1, y_centro - 2 + i],
                    [-5, 0], bordes, margenes
                )
            )

    elif opcion == 4:
        # (d) Dos grupos de boids del mismo tamaño se encuentran
        y_centro = (margenes[1][1] + margenes[1][0]) / 2
        grupo_size = 4
        boids = []
        for i in range(grupo_size):
            boids.append(
                Boid(
                    [bordes[0][0] + 1, y_centro - 3 + i*1.5],
                    [5, 0], bordes, margenes
                )
            )
            boids.append(
                Boid(
                    [bordes[0][1] - 1, y_centro - 3 + i*1.5],
                    [-5, 0], bordes, margenes
                )
            )

    elif opcion == 5:
        # (e) Dos grupos de boids de distinto tamaño se encuentran
        y_centro = (margenes[1][1] + margenes[1][0]) / 2
        grupo_izq = 3
        grupo_der = 6
        boids = []
        for i in range(grupo_izq):
            boids.append(
                Boid(
                    [bordes[0][0] + 1, y_centro - 2 + i*1.5],
                    [5, 0], bordes, margenes
                )
            )
        for i in range(grupo_der):
            boids.append(
                Boid(
                    [bordes[0][1] - 1, y_centro - 3 + i*1.0],
                    [-5, 0], bordes, margenes
                )
            )

    return boids
