import random
import os
import datetime

HEIGHT=-6
START_RANGE=10

def generate_trajectory(num_points, max_step):
    start_x = random.uniform(-START_RANGE, START_RANGE)
    start_y = random.uniform(-START_RANGE, START_RANGE)
    trajectory = [(start_x, start_y, HEIGHT)]  # Randomized starting point

    for _ in range(num_points - 1):
        last_point = trajectory[-1]
        new_point = (
            last_point[0] + random.uniform(-max_step, max_step),
            last_point[1] + random.uniform(-max_step, max_step),
            HEIGHT,  # Fixed height of 3 meters
        )
        trajectory.append(new_point)
    return trajectory

def save_trajectory(trajectory, filename):
    with open(filename, 'w') as f:
        for point in trajectory:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    num_trajectories = 5 # Number of trajectories to generate
    num_points = 20  # Points per trajectory
    max_step = 5  # Maximum step size
    # generate a run name with "airsim_traj_points" as prefix, with date_time as suffix
    output_dir = "../airsim_traj_points_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_trajectories):
        trajectory = generate_trajectory(num_points, max_step)
        filename = os.path.join(output_dir, f"trajectory_{i}.txt")
        save_trajectory(trajectory, filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    main()
