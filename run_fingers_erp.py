# run_fingers_erp.py
# This script imports the function and runs the project.

import numpy as np
from fingers_erp import calc_mean_erp


def main():
    trial_points = "events_file_ordered.csv"
    ecog_data = "brain_data_channel_one.csv"

    fingers_erp_mean = calc_mean_erp(trial_points, ecog_data)

    # Optional: save result for submission / debugging
    np.savetxt("fingers_erp_mean.csv", fingers_erp_mean, delimiter=",")

    print("Done. Output shape:", fingers_erp_mean.shape)
    print("Saved: fingers_erp_mean.csv")


if __name__ == "__main__":
    main()
