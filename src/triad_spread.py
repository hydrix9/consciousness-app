import numpy as np

def triad_spread(rA, rB, rC, o3, o6, o9, pocket_delta=10):
    """
    Given 3 RNG samples and 3 oracle instances, push them apart.
    Return their post-nudge snapshots plus triangle stats.
    """

    # Nudge each oracle differently:
    o3.nudge(rA, +pocket_delta)   # Oracle3: assert / Layer1
    o6.nudge(rB, 0)               # Oracle6: bind / Layer2
    o9.nudge(rC, -pocket_delta)   # Oracle9: prune / Layer3

    s3 = o3.snapshot_state()
    s6 = o6.snapshot_state()
    s9 = o9.snapshot_state()

    v3 = s3["vec"]
    v6 = s6["vec"]
    v9 = s9["vec"]

    # Centroid
    centroid = (v3 + v6 + v9) / 3.0

    # Edge lengths
    e36 = np.linalg.norm(v3 - v6)
    e69 = np.linalg.norm(v6 - v9)
    e93 = np.linalg.norm(v9 - v3)

    # Triangle "area" (in high dim we can approximate using 2D Heron on projections or just keep pairwise norms)
    # For now just store edge lengths + simple Heron on these magnitudes
    s = (e36 + e69 + e93) / 2.0
    approx_area = max(0.0, (s*(s-e36)*(s-e69)*(s-e93)))**0.5

    triad_info = {
        "s3": s3,
        "s6": s6,
        "s9": s9,
        "v3": v3,
        "v6": v6,
        "v9": v9,
        "centroid": centroid,
        "edges": (e36, e69, e93),
        "approx_area": approx_area,
    }

    return triad_info
