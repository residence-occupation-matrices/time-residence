import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm


def norm_pdf(x, mu, sigma):
    mu, sigma = mu.reshape(-1, 1), sigma.reshape(-1, 1)
    variance = sigma**2
    numerator = np.broadcast_to(x, (mu.shape[0], x.shape[0])) - mu
    denomenator = 2 * variance
    pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(numerator**2) / denomenator)
    return pdf


T = [0, 20, 25, 50, 55, 80, 120]
delta_a = delta_b = 28.85
n_estimators = 10
N = 100
sigma_m2 = 642

coords = np.array([[40, 0], [100, 150], [250, 250], [300, 300], [41, 1], [350, 4], [325, 0]])
x, y = np.arange(0, 400, 0.5), np.arange(0, 350, 0.5)
X, Y = np.meshgrid(x, y)

rng = np.random.default_rng()
# hz = []
hz_vec = []
for i in range(len(coords) - 1):
    a = coords[i]
    b = coords[i + 1]
    T_i = T[i + 1] - T[i]

    mc_2 = np.zeros(X.shape)

    for _ in tqdm(range(n_estimators)):
        t = rng.uniform(0, T_i, size=N)
        # alpha = t/T
        alpha = t / T_i
        mu_x = a[0] + alpha * (b[0] - a[0])
        mu_y = a[1] + alpha * (b[1] - a[1])
        sigma2 = (
            t * (1 - alpha) * sigma_m2 + (1 - alpha**2) * (delta_a**2) + (alpha**2) * (delta_b**2)
        )

        pdf_x = norm_pdf(x, mu_x, np.sqrt(sigma2))
        pdf_y = norm_pdf(y, mu_y, np.sqrt(sigma2))
        mc_2 += pdf_y.T @ pdf_x
    mc_2 *= T_i
    hz_vec.append(mc_2 / n_estimators)

hz = np.asarray(hz_vec).sum(axis=0) / (T[-1] - T[0])
print(hz.sum())
norm = cm.colors.Normalize(vmax=abs(hz).max(), vmin=-abs(hz).max())
fig, ax = plt.subplots()
cf = ax.contourf(X, Y, hz, norm=norm)
fig.colorbar(cf)
plt.show()
