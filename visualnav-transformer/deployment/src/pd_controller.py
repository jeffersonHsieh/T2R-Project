import numpy as np
import yaml
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
			 			REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle


# AIRSIM control
import airsim
from airsim.types import YawMode


# CONSTS
CONFIG_PATH = "../config/sim_drone.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]


MAX_W = robot_config["max_w"]

VEL_TOPIC = robot_config["vel_navi_topic"]
DT = 1/robot_config["frame_rate"]
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None

client = airsim.MultirotorClient()
client.confirmConnection()
# client.reset()
client.enableApiControl(True)
client.takeoffAsync().join()

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	# print("seting waypoint")
	waypoint.set(waypoint_msg.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	reached_goal = reached_goal_msg.data


def main():
	global vel_msg, reverse_mode, client
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	print("Registered with master node. Waiting for waypoints...")
	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			client.landAsync().join()
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			return
		elif waypoint.is_valid(verbose=True):
			v, w = pd_controller(waypoint.get())
			# import pdb;pdb.set_trace()
			if reverse_mode:
				v *= -1
			
			print(f"publishing new vel: {v}, {w}")

			# print(MAX_V, MAX_W)

			# yaw = YawMode(is_rate=False, yaw_or_rate=w)
			# client.moveByVelocityZBodyFrameAsync(v, 0, -3, 1,yaw_mode=yaw).join()

			# # Get the state of the drone
			state = client.getMultirotorState()

			# Extract the orientation as a quaternion
			orientation_q = state.kinematics_estimated.orientation

			# Convert quaternion to Euler angles
			orientation_euler = airsim.to_eularian_angles(orientation_q)

			# Yaw is the third element in the Euler angles (roll, pitch, yaw)
			current_yaw = orientation_euler[2]

			yaw = YawMode(is_rate=False, yaw_or_rate=clip_angle(current_yaw + w))
			vx = v * np.cos(current_yaw)
			vy = v * np.sin(current_yaw)
			client.moveByVelocityZAsync(vx, vy, -3, 1, yaw_mode=yaw).join()

			vel_msg.linear.x = v
			vel_msg.angular.z = w
			
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
