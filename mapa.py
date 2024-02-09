import matplotlib.pyplot as plt
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


biomas = {'bosque': {'mu': np.array((100, 100)),
                     'sigma': [10, 10, 0],
                     'valor': 0},
          'agua': {'mu': np.array((100, 20)),
                   'sigma': [25, 25, 0],
                   'valor': 1},
          'monte': {'mu': np.array((20, 100)),
                    'sigma': [20, 20, 0],
                    'valor': 2},
          'desert':{'mu': np.array((100, 180)),
                    'sigma': [25, 25, 0],
                    'valor': 3},
          'oscuro':{'mu': np.array((180, 100)),
                    'sigma': [20, 20, 0],
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

