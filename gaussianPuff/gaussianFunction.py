import numpy as np
from GaussianPuff.SigmaCalculation import calc_sigmas

def gauss_func_plume(Q, u, dir1, x, y, z, xs, ys, H, STABILITY):
    # Se u ~ 0: niente trasporto -> concentrazione ~ 0
    if u is None or u <= 1e-12:
        # costruisci C con shape coerente
        X, Y = (np.meshgrid(x, y, indexing='xy') if x.ndim == 1 and y.ndim == 1
                else (x, y))
        return np.zeros_like(X, dtype=float)

    # Assicuriamoci di lavorare con array
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Se arrivano 1D, creiamo la griglia 2D
    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y, indexing='xy')  # shape (ny, nx)
        Z = np.broadcast_to(z, X.shape) if np.ndim(z) == 0 else np.asarray(z)
        if Z.shape != X.shape:
            Z = np.full_like(X, float(z))  # z scalare
    else:
        X, Y = x, y
        Z = z if np.shape(z) == np.shape(X) else np.full_like(X, float(z))

    u1 = float(u)
    # componenti del vento (gradi meteo: da dove viene -> ruota di 180°)
    ang = np.deg2rad(dir1 - 180.0)
    wx = u1 * np.sin(ang)
    wy = u1 * np.cos(ang)

    # coordinate relative alla sorgente
    X1 = X - xs
    Y1 = Y - ys

    # prodotto scalare e norme per l'angolo
    dot_product = wx * X1 + wy * Y1
    r = np.sqrt(X1**2 + Y1**2)
    denom = (u1 * r) + 1e-15

    # arccos con clip per evitare NaN
    cos_theta = np.clip(dot_product / denom, -1.0, 1.0)
    subtended = np.arccos(cos_theta)

    hypotenuse = r
    downwind = np.cos(subtended) * hypotenuse
    crosswind = np.sin(subtended) * hypotenuse

    C = np.zeros_like(X, dtype=float)
    ind = downwind > 0.0

    sig_y, sig_z = calc_sigmas(STABILITY, downwind)

    C[ind] = (Q / (2.0 * np.pi * u1 * sig_y[ind] * sig_z[ind])
              * np.exp(-crosswind[ind]**2 / (2.0 * sig_y[ind]**2))
              * (np.exp(-(Z[ind] - H)**2 / (2.0 * sig_z[ind]**2))
                 + np.exp(-(Z[ind] + H)**2 / (2.0 * sig_z[ind]**2))))
    return C

def gauss_func_puff(puff, x_grid, y_grid, z_grid, dt, stability, wind_speed, wind_dir):
    # Nota: qui wind_dir non viene usata. Il centro del puff (puff.x, puff.y, puff.z)
    # è già aggiornato dallo step di advezione esterno.

    downwind_dist = max(0.0, float(wind_speed) * float(dt) * 3600.0)
    sig_y, sig_z = calc_sigmas(stability, np.array([downwind_dist]))
    sig_y = float(sig_y[0])
    sig_z = float(sig_z[0])

    # coordinate relative
    x1 = np.asarray(x_grid) - float(puff.x)
    y1 = np.asarray(y_grid) - float(puff.y)
    z1 = np.asarray(z_grid) - float(puff.z)

    # coefficiente (puff.q = massa/“quantità” del puff)
    factor = float(puff.q) / (2.0 * np.pi * sig_y * sig_z)

    C = (factor
         * np.exp(-x1**2 / (2.0 * sig_y**2))
         * np.exp(-y1**2 / (2.0 * sig_y**2))
         * (np.exp(-(z1)**2 / (2.0 * sig_z**2))
            + np.exp(-(z1 + 2.0 * float(puff.z))**2 / (2.0 * sig_z**2))))
    return C
