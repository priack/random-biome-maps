import copy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

PI = 3.141592

def norm_pdf_multivariate(x, mu, sigma):
    s1, s2, r = sigma
    x1, x2 = x - mu
    norm = s1**2 * s2**2 * (1 - r**2)
    scale = 1/(2 * PI * np.sqrt(norm))
    exp1 = -1 / (2 * norm)
    exp2 = s2**2*x1**2-2*r*s1*s2*x1*x2+s1**2*x2**2
    exp = np.exp(exp1*exp2)
    return scale * exp

sigma = [20, 20, 0]
biomas = {'bosque': {'mu': np.array((100, 100)),
                     'sigma': sigma,
                     'valor': 0},
          'agua': {'mu': np.array((100, 20)),
                   'sigma': sigma,
                   'valor': 1},
          'monte': {'mu': np.array((20, 100)),
                    'sigma': sigma,
                    'valor': 2},
          'desert':{'mu': np.array((100, 180)),
                    'sigma': sigma,
                    'valor': 3},
          'oscuro':{'mu': np.array((180, 100)),
                    'sigma': sigma,
                    'valor': 4},
          }

tamanyo = (200, 200)
mapa = np.zeros(tamanyo) - 1
p = np.zeros(len(biomas))
for i in range(200):
    for j in range(200):
        for k, b in enumerate(biomas.values()):
            p[k] = norm_pdf_multivariate(np.array([i, j]), b['mu'], b['sigma'])
        p = p / np.sum(p)
        mapa[i, j] = np.random.choice(5, p=p)

plt.imshow(mapa)
plt.show()

def elegir_paso(pos, mapa):
    d0, d1 = mapa.shape
    valido = np.zeros(4)
    vacio = np.zeros(4)
    if pos[0] + 1 != d0:
        valido[0] = 1
    if pos[0] + 1 != d0 and mapa[pos[0] + 1, pos[1]] == -1:
        vacio[0] = 1
    if pos[0] != 0:
        valido[1] = 1
    if pos[0] != 0 and mapa[pos[0] - 1, pos[1]] == -1:
        vacio[1] = 1
    if pos[1] + 1 != d0:
        valido[2] = 1
    if pos[1] + 1 != d1 and mapa[pos[0], pos[1] + 1] == -1:
        vacio[2] = 1
    if pos[1] != 0:
        valido[3] = 1
    if pos[1] != 0 and mapa[pos[0], pos[1] - 1] == -1:
        vacio[3] = 1
    if np.sum(vacio) != 0:
        p = vacio / np.sum(vacio)
        dir = np.random.choice(4,p=p)
    else:
        p = valido / np.sum(valido)
        dir = np.random.choice(4, p=p)

    if dir == 0:
        return pos[0] + 1, pos[1]
    elif dir == 1:
        return pos[0] - 1, pos[1]
    elif dir == 2:
        return pos[0], pos[1] + 1
    return pos[0], pos[1] - 1


def biomas_adyacentes(pos, mapa):
    d0, d1 = mapa.shape
    adyacentes = np.zeros(len(biomas))
    if pos[0] + 1 != d0 and mapa[pos[0] + 1, pos[1]] != -1:
        adyacentes[mapa[pos[0] + 1, pos[1]]] += 1
    if pos[0] != 0 and mapa[pos[0] - 1, pos[1]] != -1:
        adyacentes[mapa[pos[0] - 1, pos[1]]] += 1
    if pos[1] + 1 != d1 and mapa[pos[0], pos[1] + 1] != -1:
        adyacentes[mapa[pos[0], pos[1] + 1]] += 1
    if pos[1] != 0 and mapa[pos[0], pos[1] - 1] != -1:
        adyacentes[mapa[pos[0], pos[1] - 1]] += 1
    return adyacentes


# Modo paseo
nPasos = 500
longitud = 40
mapa = np.zeros(tamanyo, dtype=int) - 1
mapa[100, 100] = 0
paseo = []
pos = [100, 100]
for paso in range(nPasos):
    for stride in range(longitud):
        pos = elegir_paso(pos, mapa)
        adyacentes = biomas_adyacentes(pos, mapa)
        for k, b in enumerate(biomas.values()):
            p[k] = norm_pdf_multivariate(np.array([pos[0], pos[1]]), b['mu'], b['sigma'])
        p = p * (adyacentes + 1)**2
        p = p / np.sum(p)
        mapa[pos[0], pos[1]] = np.random.choice(5, p=p)
    paseo.append(copy.copy(mapa))

f = plt.figure()
imb = plt.imshow(paseo[0].T, vmax=4)
def animate(frame):
    imb.set_array(paseo[frame])
    imb.set_clim(vmax=4)

save_file = 'biomas_paseo.gif'
ani = FuncAnimation(f, animate, len(paseo), repeat=True, repeat_delay=1)
ani.save(save_file, writer='imagemagick', fps=60)

