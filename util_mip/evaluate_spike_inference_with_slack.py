import numpy as np

def evaluate_spike_inference_with_slack(z_est, y, tolerance_sec=0.05):
    """
    Evaluates estimated spikes against ground truth with a temporal slack window.

    Parameters:
    z_est (np.array): 1D binary array of estimated spikes (0s and 1s).
    gt_timestamps (np.array): 1D array of ground truth spike times (in seconds).
    t (np.array): 1D array of frame timestamps (in seconds).
    frame_rate (float): The sampling rate of the data (e.g., 12.175 Hz).
    tolerance_sec (float): The acceptable error margin in seconds (default 0.2s).

    Returns:
    dict: A dictionary containing TP, FP, TN, FN, and balanced measures.
    """

    gt_timestamps = y['spikes']
    t = y['t']
    frame_rate = y['frame_rate']
    # 1. Convert Ground Truth Timestamps to Binary Frame Vector
    z_gt = np.zeros_like(z_est)
    time_differences = np.abs(t[:, None] - gt_timestamps[None, :])
    nearest_frame_indices = time_differences.argmin(axis=0)
    unique_frame_indices = np.unique(nearest_frame_indices)
    z_gt[unique_frame_indices] = 1

    # 2. Calculate Slack in Frames
    # e.g., 0.2 seconds * 12.175 frames/second = 2.43 -> rounded up to 3 frames of slack
    slack_frames = int(np.ceil(tolerance_sec * frame_rate))

    # 3. Extract the exact frame indices of the spikes
    gt_idx = np.where(z_gt == 1)[0]
    est_idx = np.where(z_est == 1)[0].tolist() # Convert to list to easily 'consume' matches

    TP = 0

    # 4. Greedy Match within the Slack Window
    for g in gt_idx:
        # Find all estimated spikes that fall within [g - slack, g + slack]
        valid_matches = [e for e in est_idx if abs(e - g) <= slack_frames]

        if valid_matches:
            # If multiple exist, prioritize the one physically closest to the ground truth
            best_match = min(valid_matches, key=lambda e: abs(e - g))

            TP += 1
            # Remove the estimated spike from the pool so it isn't double-counted
            est_idx.remove(best_match)

    # 5. Calculate Remaining Confusion Matrix
    # False Negatives: Ground truth spikes that never found an estimated partner
    FN = len(gt_idx) - TP

    # False Positives: Estimated spikes left over that never matched a ground truth spike
    FP = len(est_idx)

    # True Negatives: All remaining empty frames
    TN = len(z_est) - (TP + FP + FN)

    # 6. Calculate Balanced Measures
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        'Slack_Frames': slack_frames,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1_Score': round(f1_score, 4),
        'Balanced_Accuracy': round(balanced_accuracy, 4)
    }