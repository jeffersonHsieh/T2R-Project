import airsim
import pynput
import math
import sys

import yaml
CONFIG_PATH = "../config/sim_drone.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
# original script default is 1.5

MAX_W = robot_config["max_w"] # rad/sec
DEFAULT_YAW_RATE = 50 # math.degrees(MAX_W) # deg/sec
# original script default is 50 deg/s * 0.2 secs

def init_flight(client: airsim.MultirotorClient):
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    # get height
    _,_,z = get_drone_position(client)
    print(f"Height: {z}")
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0,0,z), airsim.to_quaternion(0, 0, 0)), True)
    print("Ready to fly!")

def get_drone_position(client: airsim.MultirotorClient):
    pos = client.simGetObjectPose(client.listVehicles()[0]).position
    return [pos.x_val, pos.y_val, pos.z_val]

def get_drone_orientation(client: airsim.MultirotorClient):
    orientation = client.simGetObjectPose(client.listVehicles()[0]).orientation
    return quaternion_to_euler_angles(orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)

def on_press(key: pynput.keyboard.Key, client: airsim.MultirotorClient, step: float = 1, velocity: float = MAX_V, yaw_rate: float = DEFAULT_YAW_RATE, yaw_duration: float = 0.2):
    if not isinstance(key, pynput.keyboard.Key):
        x, y, z = get_drone_position(client)
        yaw, pitch, roll = get_drone_orientation(client)
        print(f"xyz {x, y, z}")
        if key.char == 's':
            #client.moveToPositionAsync(x - step, y, z, velocity)
            client.moveToPositionAsync(x - step * math.cos(yaw), y - step * math.sin(yaw), z, velocity)
        if key.char == 'w':
            #client.moveToPositionAsync(x + step, y, z, velocity)
            client.moveToPositionAsync(x + step * math.cos(yaw), y + step * math.sin(yaw), z, velocity)
        if key.char == 'd':
            #client.moveToPositionAsync(x, y - step, z, velocity)
            client.moveToPositionAsync(x - step * math.sin(yaw), y + step * math.cos(yaw), z, velocity)
        if key.char == 'a':
            #client.moveToPositionAsync(x, y + step, z, velocity)
            client.moveToPositionAsync(x + step * math.sin(yaw), y - step * math.cos(yaw), z, velocity)
        if key.char == 'e':
            client.rotateByYawRateAsync(yaw_rate, yaw_duration)
        if key.char == 'q':
            client.rotateByYawRateAsync(-yaw_rate, yaw_duration)
        if key.char == 'z':
            client.moveToPositionAsync(x, y, z - step * 1.1, velocity)
        if key.char == 'x':
            client.moveToPositionAsync(x, y, z + step, velocity)
        if key.char == 'c':
            print('killing gracefully...')
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0,0,0), airsim.to_quaternion(0, 0, 0)), True)
            client.armDisarm(False)
            client.enableApiControl(False)
            sys.exit()
        #     client.reset()
        #     sys.exit()


def control_loop(client: airsim.MultirotorClient):
    with pynput.keyboard.Listener(on_press=lambda event: on_press(event, client=client)) as listener:
        listener.join()
            
#function that converts quaternions to euler angles
def quaternion_to_euler_angles(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]

def main():
    client = airsim.MultirotorClient(ip="172.18.80.1")
    # client.reset()
    # try:
    init_flight(client)
    control_loop(client)
        # client.reset()
    # finally:
        # print('killing gracefully...')
        # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0,0,0), airsim.to_quaternion(0, 0, 0)), True)
        # client.armDisarm(False)
        # client.enableApiControl(False)

if __name__ == "__main__":
    main()