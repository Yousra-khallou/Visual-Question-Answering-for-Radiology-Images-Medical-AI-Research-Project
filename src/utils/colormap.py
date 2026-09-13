import numpy as np

def plt_jet(x: np.ndarray) -> np.ndarray:
    """
    Colormap 'jet' purement vectorisée en NumPy sans dépendance matplotlib.
    Entrée: tableau normalisé entre 0 et 1.
    Sortie: tableau RGB float [H, W, 3] normalisé entre 0 et 1.
    """
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)
