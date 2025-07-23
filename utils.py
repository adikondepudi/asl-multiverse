# utils.py
import numpy as np

def engineer_signal_features(raw_signal: np.ndarray, num_plds: int) -> np.ndarray:
    """
    Engineers explicit shape-based features from raw ASL signal curves to
    make timing information more salient for the neural network.

    Args:
        raw_signal: A numpy array of shape (N, num_plds * 2) or (num_plds * 2,)
                    containing concatenated PCASL and VSASL signals.
        num_plds: The number of Post-Labeling Delays per modality.

    Returns:
        A numpy array of shape (N, 4) with the engineered features:
        - PCASL time-to-peak index
        - VSASL time-to-peak index
        - PCASL center-of-mass
        - VSASL center-of-mass
    """
    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(1, -1) # Ensure 2D for processing

    num_samples = raw_signal.shape[0]
    engineered_features = np.zeros((num_samples, 4))
    plds_indices = np.arange(num_plds)

    for i in range(num_samples):
        pcasl_curve = raw_signal[i, :num_plds]
        vsasl_curve = raw_signal[i, num_plds:]

        # Feature 1: Time to peak (index of max signal)
        engineered_features[i, 0] = np.argmax(pcasl_curve)
        engineered_features[i, 1] = np.argmax(vsasl_curve)

        # Feature 2: Center of mass (temporal)
        pcasl_sum = np.sum(pcasl_curve) + 1e-6
        vsasl_sum = np.sum(vsasl_curve) + 1e-6
        engineered_features[i, 2] = np.sum(pcasl_curve * plds_indices) / pcasl_sum
        engineered_features[i, 3] = np.sum(vsasl_curve * plds_indices) / vsasl_sum

    return engineered_features.astype(np.float32)