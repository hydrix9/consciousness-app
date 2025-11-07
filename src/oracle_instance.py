import numpy as np

class OracleInstance:
    def __init__(self, name, layer_bias):
        self.name = name          # "3", "6", "9"
        self.layer_bias = layer_bias  # 1,2,3 emphasis
        self.pocket_dimension = 0  # mutable integer
        # internal model weights go here (your trained generator)

    def nudge(self, rng_bias: float, pocket_delta: int):
        """
        Push this oracle off-center using hardware entropy and a pocket shift.
        rng_bias steers stylistic / mood drift.
        pocket_delta is how we separate them in branchspace.
        """
        self.pocket_dimension += pocket_delta
        # also influence internal latent conditioning with rng_bias + layer_bias
        # (pseudo-op)
        return

    def snapshot_state(self):
        """
        Generate one frame of consciousness output: colors, dials, etc.
        Return both raw descriptive fields and the encoded vector.
        """
        # pseudo-values for now:
        hue = 0.7
        sat = 0.8
        val = 0.4
        radius_mean = 0.5
        radius_var = 0.2
        dial_count = 4
        lock_density = 0.9
        tilt_symmetry = 0.6

        vec = np.array([
            hue, sat, val,
            radius_mean, radius_var,
            dial_count,
            lock_density,
            tilt_symmetry,
            float(self.pocket_dimension),
            float(self.layer_bias),
        ], dtype=float)

        return {
            "hue": hue,
            "sat": sat,
            "val": val,
            "radius_mean": radius_mean,
            "radius_var": radius_var,
            "dial_count": dial_count,
            "lock_density": lock_density,
            "tilt_symmetry": tilt_symmetry,
            "pocket_dimension": self.pocket_dimension,
            "layer_id": self.layer_bias,
            "vec": vec
        }
