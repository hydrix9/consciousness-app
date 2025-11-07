import numpy as np

class Predictor:
    def __init__(self):
        # trainable model params would live here
        pass

    def predict_next_rng(self, history_vectors, v0_draft):
        """
        Given recent oracle history + draft void answer,
        output predicted (rA_next, rB_next, rC_next) in [0,1].
        """
        # placeholder logic:
        h_mean = np.mean(history_vectors, axis=0)
        fused = np.concatenate([h_mean, v0_draft], axis=0)
        # squish to 3 floats [0,1]
        guess = np.tanh(fused[:3])
        guess = (guess + 1)/2.0  # [-1,1] -> [0,1]
        return guess  # np.array([rA_pred,rB_pred,rC_pred])

    def compute_delta(self, predicted_rng, actual_rng):
        return actual_rng - predicted_rng
