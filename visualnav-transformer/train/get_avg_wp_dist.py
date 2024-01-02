import os
import numpy as np
import pickle

def calculate_average_distance(positions):
    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
    return np.mean(distances)

def main(dataset_path):
    total_average_distance = 0
    trajectory_count = 0

    for traj_name in os.listdir(dataset_path):
        traj_path = os.path.join(dataset_path, traj_name)

        if os.path.isdir(traj_path):
            traj_data_path = os.path.join(traj_path, "traj_data.pkl")

            with open(traj_data_path, 'rb') as file:
                traj_data = pickle.load(file)

            average_distance = calculate_average_distance(traj_data['position'])
            total_average_distance += average_distance
            trajectory_count += 1

    overall_average = total_average_distance / trajectory_count
    return overall_average

# Use the function

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        help="path of the dataset",
        required=True,
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    overall_average_distance = main(dataset_path)
    print(f"The overall average distance is: {overall_average_distance}")
