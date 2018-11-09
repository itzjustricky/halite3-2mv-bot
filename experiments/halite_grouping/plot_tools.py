"""

Some utilities for plotting.
"""

import numpy as np
import matplotlib.cm as cm


def generate_colors(n_colors: int) -> np.ndarray:
    """ Generate some colors in the form a matrix. Colors are
        represented by 4 numbers.
    """
    return cm.rainbow(np.linspace(0, 1, n_colors))
