import matplotlib.pyplot as plt
import numpy as np


def det2_2(m):
    return m[0, 0] * m[1, 1] - m[0, 1]*m[1, 0]

def inv2_2(m):
    det = det2_2(m)
    inv = np.array([[m[1,1], -m[0, 1]], [-m[1,0], m[0, 0]]])
    return inv / det

def norm_pdf_multivariate(x, mu, sigma):
    det = det2_2(sigma)
    size = len(x)
    if det == 0:
        raise NameError("The covariance matrix can't be singular")
    norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.power(det, 1 / 2))
    x_mu = x - mu
    inv = inv2_2(sigma)
    # Mirar como simplificar la multiplicacion de vector y matriz
    result = np.exp(-0.5 * (x_mu.dot(inv).dot(x_mu.T)))
    return norm_const * result


biomas = {'bosque': {'mu': np.array((100, 100)),
                     'sigma': np.array([[5000, 0], [0, 5000]]),
                     'valor': 0},
          'agua': {'mu': np.array((100, 20)),
                   'sigma': np.array([[5000, 0], [0, 5000]]),
                   'valor': 1},
          'monte': {'mu': np.array((20, 100)),
                    'sigma': np.array([[5000, 0], [0, 5000]]),
                    'valor': 2},
          'desert':{'mu': np.array((100, 180)),
                    'sigma': np.array([[5000, 0], [0, 5000]]),
                    'valor': 3},
          'oscuro':{'mu': np.array((180, 100)),
                    'sigma': np.array([[5000, 0], [0, 5000]]),
                    'valor': 4},
          }

tamanyo = (200, 200)
mapa = np.zeros(tamanyo)
p = np.zeros(len(biomas))
for i in range(200):
    for j in range(200):
        for k, b in enumerate(biomas.values()):
            p[k] = norm_pdf_multivariate(np.array([i, j]), b['mu'], b['sigma'])
        p = p / np.sum(p)
        mapa[i, j] = np.random.choice(5, p=p)

plt.imshow(mapa)
plt.show()

