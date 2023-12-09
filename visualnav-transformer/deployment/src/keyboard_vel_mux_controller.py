import pynput
import math
import sys
import rospy
import yaml

from geometry_msgs.msg import Twist



CONFIG_PATH = "../config/sim_drone.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
# original script default is 1.5

MAX_W = robot_config["max_w"] # rad/sec
DEFAULT_YAW_RATE = 50 # math.degrees(MAX_W) # deg/sec
# original script default is 50 deg/s * 0.2 secs
if "set_height" in robot_config:
	HEIGHT = robot_config["set_height"]
	if HEIGHT > 0:
		HEIGHT*=-1
else:
	HEIGHT = -3

VEL_TELEOP_TOPIC = robot_config["vel_teleop_topic"]


def on_press(key: pynput.keyboard.Key, pub:rospy.Publisher , step: float = 1, velocity: float = MAX_V, yaw_rate: float = DEFAULT_YAW_RATE, yaw_duration: float = 0.2):
    if not isinstance(key, pynput.keyboard.Key):
        # x, y, z = get_drone_position(client)
        # yaw, pitch, roll = get_drone_orientation(client)
        twist = Twist()
        if key.char == 's':
             # backward
            twist.linear.x = -velocity
        if key.char == 'w':
            # forward
            twist.linear.x = velocity

        if key.char == 'd':
            #  right
            twist.linear.y = velocity
        if key.char == 'a':
            # left
            twist.linear.y = -velocity

        if key.char == 'e':
            twist.angular.z = yaw_rate
            # rotate right
        if key.char == 'q':
            twist.angular.z = -yaw_rate
            # rotate left
        if key.char == 'z':
            twist.linear.z = velocity
            # raise altitude
        if key.char == 'x':
            # lower altitude
            twist.linear.z = -velocity

        if key.char == 'c':
            sys.exit()
        
        pub.publish(twist)



def control_loop(pub: rospy.Publisher):
    with pynput.keyboard.Listener(on_press=lambda event: on_press(event,pub)) as listener:
        listener.join()


def main():
    rospy.init_node("TELEOP", anonymous=False)
    vel_out = rospy.Publisher(VEL_TELEOP_TOPIC, Twist, queue_size=1)
    control_loop(vel_out)


if __name__ == "__main__":
    main()