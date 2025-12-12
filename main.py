# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    if not isinstance(n, int) or n <= 0:
        return None
    if n == 1:
        return np.array([1.0])
    k = np.arange(n)
    return np.cos(np.pi * k / (n - 1))

def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    w = np.zeros(n)

    if not isinstance(n, int) or n<=0:
        return None
    for j in range(n):
        if j == 0 or j == n-1:
            w[j] = ((-1)**j)*1/2
        else:
            w[j]=((-1)**j)*1
    return w


def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray) or not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray):
        return None
    p = np.zeros_like(x, dtype=float)

    for j, xj in enumerate(x):
        diff = xj - xi

        if np.any(diff == 0):
            p[j] = yi[diff == 0][0]
        else:
            numerator = np.sum(wi * yi / diff)
            denominator = np.sum(wi / diff)
            p[j] = numerator / denominator

    return p


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    return np.max(np.abs(np.array(xr)-np.array(x)))
