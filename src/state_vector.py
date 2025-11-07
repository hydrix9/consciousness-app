import numpy as np

def encode_oracle_state(hue, sat, val,
                        radius_mean, radius_var,
                        dial_count, lock_density,
                        tilt_symmetry,
                        pocket_dimension,
                        layer_id):
    """
    Flatten relevant oracle output into a single numeric vector.
    All inputs should be normalized or normalizable.
    """
    return np.array([
        hue, sat, val,
        radius_mean, radius_var,
        dial_count,
        lock_density,
        tilt_symmetry,
        pocket_dimension,
        layer_id
    ], dtype=float)
