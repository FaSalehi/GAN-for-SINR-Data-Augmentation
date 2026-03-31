import numpy as np
from scipy.spatial.distance import jensenshannon

def compute_jsd(real_samples, generated_samples, bins=100):
    # Histogram the real and generated data
    p_hist, bin_edges = np.histogram(real_samples, bins=bins, range=(0, 1), density=False)
    q_hist, _ = np.histogram(generated_samples, bins=bins, range=(0, 1), density=False)

    # Normalize to form proper probability distributions
    # p_hist += 1e-8  # prevent zero division
    # q_hist += 1e-8
    # p_hist /= p_hist.sum()
    # q_hist /= q_hist.sum()

    # Compute Jensen-Shannon divergence
    jsd = jensenshannon(p_hist, q_hist, base=2.0)
    return jsd