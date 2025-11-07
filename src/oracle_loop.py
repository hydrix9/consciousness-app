import numpy as np
from rng import RNGSource
from oracle_instance import OracleInstance
from triad_spread import triad_spread
from oracle_zero import OracleZero
from predictor import Predictor
from memory import log_experiment

def hash_question_to_signature(qtext: str) -> np.ndarray:
    """
    Step 0. Ask a question -> signature.
    Could also be EEG gesture summary.
    """
    # super naive hash -> 4 floats
    h = abs(hash(qtext))
    floats = [
        ((h >>  0) & 0xff) / 255.0,
        ((h >>  8) & 0xff) / 255.0,
        ((h >> 16) & 0xff) / 255.0,
        ((h >> 24) & 0xff) / 255.0,
    ]
    return np.array(floats, dtype=float)

def run_one_query(question_text: str,
                  rng_dev="/dev/TrueRNG0",
                  history_buffer=[],
                  log_path="oracle_runs.jsonl"):
    """
    Execute Steps 0-6.
    """

    # --- init hardware / models ---
    rng = RNGSource(rng_dev)
    o3 = OracleInstance("3", layer_bias=1)
    o6 = OracleInstance("6", layer_bias=2)
    o9 = OracleInstance("9", layer_bias=3)
    o0 = OracleZero()
    pred = Predictor()

    # Step 0. question signature
    qsig = hash_question_to_signature(question_text)

    # Step 1. triad spread
    rA, rB, rC = rng.triple_sample()
    triad_info = triad_spread(rA, rB, rC, o3, o6, o9, pocket_delta=10)

    # We'll treat recent triad snapshots as "history" for Predictor.
    # For now, just push this triad_info centroid etc onto history_buffer.
    history_vec = triad_info["centroid"]
    history_buffer.append(history_vec)
    if len(history_buffer) > 32:
        history_buffer = history_buffer[-32:]

    # Step 2. void synthesis
    v0_pkg = o0.synthesize(triad_info)
    v0_draft = v0_pkg["v0_draft"]

    # Step 3. predict reality
    predicted_rng = pred.predict_next_rng(history_buffer, v0_draft)

    # Step 4. observe reality
    rA_next, rB_next, rC_next = rng.triple_sample()
    actual_rng = np.array([rA_next, rB_next, rC_next], dtype=float)

    delta_vec = pred.compute_delta(predicted_rng, actual_rng)

    # Upgrade C: recursive correction
    v0_corrected = o0.refine_with_delta(v0_draft, delta_vec)

    # Step 5. interpret Δ
    # We don't do English here. Interpretation is down-stream clustering of delta_vec.

    # Step 6. learn / log
    log_experiment(
        log_path,
        qsig,
        v0_draft,
        predicted_rng,
        actual_rng,
        delta_vec,
        v0_corrected=v0_corrected
    )

    return {
        "question_sig": qsig,
        "triad_info": triad_info,
        "v0_draft": v0_draft,
        "predicted_rng": predicted_rng,
        "actual_rng": actual_rng,
        "delta_vec": delta_vec,
        "v0_corrected": v0_corrected
    }
