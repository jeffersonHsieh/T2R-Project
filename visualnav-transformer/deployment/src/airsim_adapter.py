#!/usr/bin/env python
"""
Interface between airsim and VINT/teleop/etc controls
Subscribes to /cmd_vel_out, which is the output of the twist_mux from multiple controllers
and interact with airsim client to move the drone
"""

import rospy
import airsim
from geometry_msgs.msg import Twist
from airsim.types import YawMode
import yaml
import argparse

# CONSTS
# all velocity commands are funneled through twsit_mux, prioritized, and published on this topic
# check config/twist_mux.yaml for info
VEL_TOPIC="/cmd_vel_out"
vel_msg = Twist()

CONFIG_PATH = "../config/sim_drone.yaml"
with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

DT = 1/robot_config["frame_rate"]

if "set_height" in robot_config:
    HEIGHT = robot_config["set_height"]
    if HEIGHT > 0:
        HEIGHT*=-1
else:
    HEIGHT = -3

RATE = robot_config["frame_rate"]


def twist_callback(msg):
    # Translate Twist message to AirSim commands
    # Example: client.moveByVelocityZBodyFrameAsync(msg.linear.x, 0, HEIGHT, DT, ...)
    # get the linear and angular velocity from the twist message
    linear_vel = msg.linear.x # m/s
    angular_vel = msg.angular.z # deg/s
    print("received message: ", linear_vel, angular_vel)
    yaw = YawMode(is_rate=True, yaw_or_rate=float(angular_vel))
    client.moveByVelocityZBodyFrameAsync(float(linear_vel), float(0), HEIGHT, DT,yaw_mode=yaw,drivetrain=airsim.DrivetrainType.ForwardOnly).join()

def check_for_collision(client):
	collision_info = client.simGetCollisionInfo()
	return collision_info.has_collided

def set_initial_position(client, position):
	client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(position[0], position[1], position[2]), airsim.to_quaternion(0, 0, 0)), True)


def main(args,client):
    global vel_msg
    rospy.init_node("AIRSIM_INTERFACE", anonymous=False)
    # Initialize AirSim client

    client.confirmConnection()
    client.enableApiControl(True)
    client.takeoffAsync().join()
    twist_sub = rospy.Subscriber(VEL_TOPIC, Twist, twist_callback)

    # set initial position
    set_initial_position(client, args.initial_position)
    rospy.spin()
    # rate=rospy.Rate(RATE)
    # while not rospy.is_shutdown():
    #     rate.sleep()
        # check for collision
        # if check_for_collision(client):
        #     print("Collision detected!")
        #     break
    client.reset()
    # set_initial_position(client, [0,0,0])
    # client.landAsync().join()

if __name__ == '__main__':
    # add argument to set initial position


    parser = argparse.ArgumentParser()
    # short flag "-i", long flag "--initial_position
    parser.add_argument('-i', '--initial_position', nargs='+', type=float, default=[0,0,HEIGHT])    

    args = parser.parse_args()
    HOST = '127.0.0.1' # Standard loopback interface address (localhost)
    from platform import uname
    import os
    if 'linux' in uname().system.lower() and 'microsoft' in uname().release.lower(): # In WSL2
        if 'WSL_HOST_IP' in os.environ:
            HOST = os.environ['WSL_HOST_IP']
            print("Using WSL2 Host IP address: ", HOST)
        client = airsim.MultirotorClient(ip=HOST)
    else:
        client = airsim.MultirotorClient()
    main(args,client)