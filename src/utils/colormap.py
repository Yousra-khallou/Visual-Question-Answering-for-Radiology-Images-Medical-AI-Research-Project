import numpy as np

def plt_jet(x: np.ndarray) -> np.ndarray:
    """
    Vectorized NumPy implementation of the 'jet' colormap without matplotlib dependency.
    Input : Normalized array in range [0, 1].
    Output: RGB float array [H, W, 3] in range [0, 1].
    """
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)
