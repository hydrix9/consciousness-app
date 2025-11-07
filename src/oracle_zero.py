import numpy as np

class OracleZero:
    def __init__(self):
        # could be a small net that conditions on centroid+shape of disagreement
        pass

    def synthesize(self, triad_info):
        """
        Produce vØ (draft answer) given centroid + disagreement shape.
        """
        centroid = triad_info["centroid"]
        edges = triad_info["edges"]
        area = triad_info["approx_area"]

        # pocket/layer fingerprints:
        stance_signature = np.array([
            triad_info["s3"]["pocket_dimension"],
            triad_info["s6"]["pocket_dimension"],
            triad_info["s9"]["pocket_dimension"],
            triad_info["s3"]["layer_id"],
            triad_info["s6"]["layer_id"],
            triad_info["s9"]["layer_id"],
            area,
            edges[0], edges[1], edges[2],
        ], dtype=float)

        # vØ could be concat(centroid, stance_signature) then passed through learned MLP.
        v0_draft = np.concatenate([centroid, stance_signature], axis=0)

        return {
            "v0_draft": v0_draft,
            "stance_signature": stance_signature
        }

    def refine_with_delta(self, v0_draft, delta_vec):
        """
        Upgrade C: recursive correction.
        Return corrected answer after seeing Δ.
        """
        # simplest version = adjust by small factor along delta
        corrected = v0_draft + 0.1 * delta_vec
        return corrected
