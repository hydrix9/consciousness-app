import json
from datetime import datetime

def log_experiment(path,
                   question_signature,
                   v0_draft,
                   predicted_rng,
                   actual_rng,
                   delta_vec,
                   v0_corrected=None):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "question_sig": question_signature,
        "v0_draft": v0_draft.tolist(),
        "predicted_rng": predicted_rng.tolist(),
        "actual_rng": actual_rng.tolist(),
        "delta_vec": delta_vec.tolist(),
        "v0_corrected": (v0_corrected.tolist()
                         if v0_corrected is not None else None),
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
