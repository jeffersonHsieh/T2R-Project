import airsim
import time
import math
import sys
from pathlib import Path
import yaml

CONFIG_PATH = "../config/sim_drone.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]


MAX_W = robot_config["max_w"]

def read_trajectory(filename:Path):
    trajectory = []
    with open(str(filename), 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            trajectory.append((x, y, z))
    return trajectory


def set_initial_position(client, position):
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]), airsim.to_quaternion(0, 0, 0)), True)

def calculate_yaw_angle(current_point, next_point):
    """
    Calculate the yaw angle (in radians) required to face the next point.
    :param current_point: Tuple (x, y, z) representing the current position.
    :param next_point: Tuple (x, y, z) representing the next position.
    :return: Yaw angle in radians.
    """
    delta_x = next_point[0] - current_point[0]
    delta_y = next_point[1] - current_point[1]
    yaw_angle = math.atan2(delta_y, delta_x)  # Angle in radians
    return yaw_angle


def check_for_collision(client):
    collision_info = client.simGetCollisionInfo()
    return collision_info.has_collided

def follow_trajectory(client, trajectory, velocity):
    set_initial_position(client, trajectory[0])  # Set initial position to the first point of the trajectory
    client.takeoffAsync().join()  # Take off
    time.sleep(1)  # Wait a bit after takeoff
    
    for i in range(len(trajectory) - 1):
        current_point = trajectory[i]
        next_point = trajectory[i + 1]
        yaw_angle = calculate_yaw_angle(current_point, next_point)

        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(yaw_angle))
        print("tracking pos, yaw = ", next_point, math.degrees(yaw_angle))
        # client.moveToPositionAsync(*next_point, velocity, yaw_mode=yaw_mode).join()
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*next_point), airsim.to_quaternion(0, 0, 0)), True)
        time.sleep(1)  # Wait for stability

        # if check_for_collision(client):
        #     print(f"Collision detected at point {i} in the trajectory.")
        #     return i  # Return the index of the point where collision occurred

    return -1  # Return -1 if no collision occurred

def main():
    if len(sys.argv) != 2:
        print("Usage: python track_trajectory.py <path_to_trajectory_file>")
        sys.exit(1)

    trajectory_file = Path(sys.argv[1])
    # log file is at the same directory as the trajectory file, with the name prefixed with directory name
    track_log = trajectory_file.parent / (trajectory_file.parent.name + "_track_log.csv")
    HOST = '127.0.0.1' # Standard loopback interface address (localhost)
    from platform import uname
    import os
    if 'linux' in uname().system.lower() and 'microsoft' in uname().release.lower(): # In WSL2
        if 'WSL_HOST_IP' in os.environ:
            HOST = os.environ['WSL_HOST_IP']
            print("Using WSL2 Host IP address: ", HOST)
    client = airsim.MultirotorClient(ip=HOST)
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    trajectory = read_trajectory(trajectory_file)
    collision_point = follow_trajectory(client, trajectory, velocity=MAX_V)
    if collision_point != -1:
        print(f"Trajectory terminated due to collision at point {collision_point}.")
    else:
        print("Trajectory completed without collision.")
    
    # append one row to traj log with the following info:
    # traj_name, collision_point_index, collision_point_x, collision_point_y, collision_point_z
    # if no collision, then collision_point_index = -1 and the rest are 0
    # if collision, then collision_point_index is the index of the collision point in the trajectory
    # and the rest are the x, y, z of the collision point
    with open(str(track_log), 'a') as f:
        if collision_point == -1:
            f.write(f"{trajectory_file},-1,0,0,0\n")
        else:
            collision_point_x = trajectory[collision_point][0]
            collision_point_y = trajectory[collision_point][1]
            collision_point_z = trajectory[collision_point][2]
            f.write(f"{trajectory_file},{collision_point},{collision_point_x},{collision_point_y},{collision_point_z}\n")



    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
