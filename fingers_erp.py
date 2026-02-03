import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_mean_erp(trial_points, ecog_data):
    """
    Calculate mean ERP for each finger movement.

    Parameters
    ----------
    trial_points : str
        Path to CSV file with columns:
        starting_point, peak_point, finger
        (must be treated as integers)
    ecog_data : str
        Path to CSV file with one column (ECoG signal)

    Returns
    -------
    fingers_erp_mean : np.ndarray
        Array of shape (5, 1201):
        mean ERP for fingers 1–5
    """

    # ---------- Load trial points (WITH header) ----------
    tp = pd.read_csv(trial_points)

    # Keep only the first 3 columns (start, peak, finger)
    tp = tp.iloc[:, :3]

    # Convert to numeric, drop bad rows, enforce int
    tp = tp.apply(pd.to_numeric, errors="coerce")
    tp = tp.dropna()
    tp = tp.astype(int)

    tp.columns = ["start_idx", "peak_idx", "finger"]

    # ---------- Load ECoG signal ----------
    ecog = pd.read_csv(ecog_data, header=None)
    signal = ecog.iloc[:, 0].to_numpy(dtype=float)

    # ---------- ERP window definition ----------
    pre = 200     # samples before start
    post = 1000   # samples after start
    win_len = pre + 1 + post  # 1201

    # Container for epochs per finger
    finger_epochs = {f: [] for f in range(1, 6)}
    signal_len = len(signal)

    # ---------- Extract epochs ----------
    for _, row in tp.iterrows():
        start = row["start_idx"]
        finger = row["finger"]

        if finger not in finger_epochs:
            continue

        left = start - pre
        right = start + post

        # Skip if window exceeds signal bounds
        if left < 0 or right >= signal_len:
            continue

        epoch = signal[left:right + 1]

        if len(epoch) == win_len:
            finger_epochs[finger].append(epoch)

    # ---------- Compute mean ERP ----------
    fingers_erp_mean = np.full((5, win_len), np.nan)

    for i, finger in enumerate(range(1, 6)):
        if len(finger_epochs[finger]) > 0:
            fingers_erp_mean[i, :] = np.mean(
                finger_epochs[finger], axis=0
            )

    # ---------- Plot ----------
    time = np.arange(-pre, post + 1)

    plt.figure(figsize=(8, 5))
    for i, finger in enumerate(range(1, 6)):
        plt.plot(time, fingers_erp_mean[i], label=f"Finger {finger}")

    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("Time (ms relative to movement start)")
    plt.ylabel("Mean ECoG signal")
    plt.title("Mean ERP per Finger")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fingers_erp_mean

