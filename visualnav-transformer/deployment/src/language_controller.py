"""
This controller takes in a natural language input
and outputs a velocity command to the robot (linear and angular)
The velocity message is multiplexed with other controllers 
using the cmd_vel_mux node

one limitation is this architecture won't allow us to use cost maps
since we are only outputting linear and angular
a next step would be to multiplex at the waypoint publishing level instead
and have a cost map controller that outputs a waypoint

currently, we've only implemented open loop control
closed loop control might be to subscribe to the odometry topic and track desired yaw rotation

You should launch this under the GroundingDino environment
"""

# ROS
import rospy
from geometry_msgs.msg import Twist
import yaml
from sensor_msgs.msg import Image, CompressedImage
# import string message
from std_msgs.msg import String

from torchvision.ops import box_convert
import torch
import cv2
import numpy as np


from PIL import Image as PILImage
import io
from ros_data import ROSData
# CONSTS
CONFIG_PATH = "../config/sim_drone.yaml"
# use datetime to create a unique log directory
import datetime
now = datetime.datetime.now()
LOGS_PATH = f"../logs/airsim_explore_{now.strftime('%Y-%m-%d_%H-%M-%S')}/"
# create logs directory if it doesn't exist
import os
if not os.path.exists(LOGS_PATH):
	os.makedirs(LOGS_PATH)
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]

BASE_YAW_RATE=30 # deg/sec

if "set_height" in robot_config:
	HEIGHT = robot_config["set_height"]
	if HEIGHT > 0:
		HEIGHT*=-1
else:
	HEIGHT = -3

RATE=robot_config["frame_rate"]

# subscribe-to topics, for closed loop feedback
IMAGE_TOPIC = "/airsim_node/SimpleFlight/front_center/Scene/compressed"
TEXT_TOPIC = "/text_input"
ODOM_TOPIC = "/airsim_node/SimpleFlight/odom_local_ned"

# publish-to topics
VEL_LANG_TOPIC = robot_config["vel_lang_topic"]
print("vel topic: ", VEL_LANG_TOPIC)
LANG_TIMEOUT = 1 # seconds


import os
from pathlib import Path
# use realpath for Path
HOME=Path(__file__).resolve().parent.parent.parent.parent/"GroundingDINO"
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import supervision as sv

DINO_CONFIG_PATH=os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")

model = load_model(DINO_CONFIG_PATH, WEIGHTS_PATH)
# import pdb;pdb.set_trace()


TEXT_PROMPT = "tree,house"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# might need a cold run to get it onto the GPU
# IMAGE_DIR="/home/jefferson/Desktop/T2R-Project/visualnav-transformer/datasets/airsim_explore/bags_airsim_explore_2023-12-03-19-39-54_0/"
# IMAGE_PATH=os.path.join(IMAGE_DIR, "157.jpg")
# image_source, image = load_image(IMAGE_PATH)

# boxes, logits, phrases = predict(
# 	model=model,
# 	image=image,
# 	caption=TEXT_PROMPT,
# 	box_threshold=BOX_TRESHOLD,
# 	text_threshold=TEXT_TRESHOLD
# )

# h, w, _ = image_source.shape
# boxes = boxes * torch.Tensor([w, h, w, h]) # unnormalized the cx,cy,w,h format
# # xyxy=box_convert(boxes,in_fmt="cxcywh", out_fmt="xyxy").numpy()
# # print(boxes)

def compressed_msg_to_pil(msg: CompressedImage) -> PILImage.Image:
	# The data is compressed, so we need to decompress it first
	pil_image = PILImage.open(io.BytesIO(msg.data))
	return pil_image

obs_img=ROSData(0.1, name="obs_img")

def callback_obs(msg):
	global obs_img
	# print("recieved image")
	obs_img.set(compressed_msg_to_pil(msg))

text_input_msg=ROSData(LANG_TIMEOUT, name="text_input")
def callback_text(msg):
	global text_input_msg
	text_input_msg.set(msg.data)

def transform_img(image_source):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = np.asarray(image_source.convert("RGB"))
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def predict_boxes(image, text_prompt, box_threshold, text_threshold):
	boxes, logits, phrases = predict(
		model=model,
		image=image,
		caption=text_prompt,
		box_threshold=box_threshold,
		text_threshold=text_threshold
	)

	return boxes, logits, phrases

suffix_idx=0
def draw_boxes_to_log(image_source, bboxes, logits, phrases, log_path):
	# assumes normalized boxes
	# assume cx,cy,w,h format
	annotated_frame = annotate(
		image_source=image_source,
		boxes=bboxes,
		logits=logits,
		phrases=phrases,
	)
	cv2.imwrite(log_path, annotated_frame)

def simple_parser(text_input):
	"""Parse text input into a list of objects"""
	mode = -1 if "avoid" in text_input else 1
	objects = text_input.split(",")
	return mode, ",".join(objects[1:])

def navigation_policy(egocentric_image, text_input, parsing_function, bounding_box_function, proximity_threshold, center_threshold, base_yaw_rate):
	# Get the bounding box and classification
	mode, prompt = parsing_function(text_input)
	image_source, egocentric_image = transform_img(egocentric_image)

	bboxes,logits,phrases = bounding_box_function(egocentric_image, prompt,BOX_TRESHOLD,TEXT_TRESHOLD)  # Returns [cx, cy, w, h]
	print("mode:", mode, "; prompt:", prompt, "; bboxes:", bboxes)
	# Image center (assuming image dimensions are known)
	image_center_x = egocentric_image.shape[1] / 2

	# rescale
	h, w, _ = egocentric_image.shape
	boxes = bboxes * torch.Tensor([w, h, w, h]) # unnormalized the cx,cy,w,h format

	# Bounding box center and size
	# pick the bbox with the largest area
	yaw_rate = 0

	if len(boxes) > 0:
		print("detected bboxes")
		bounding_box = boxes[0]
		bbox_center_x, bbox_width = bounding_box[0], bounding_box[2]
		global suffix_idx
		suffix_idx+=1

		#use normalized boxes
		draw_boxes_to_log(image_source, bboxes, logits, phrases, LOGS_PATH+f"log_{suffix_idx}.jpg")
		# Calculate yaw rate
		toward_yaw_rate = base_yaw_rate
		if mode==-1:
			if abs(bbox_center_x - image_center_x) <= proximity_threshold:
				# Scale yaw rate based on bounding box size
				scale_factor = max(1, bbox_width / egocentric_image.shape[1])  # Scale factor based on bbox width relative to image width
				if bbox_center_x < image_center_x - center_threshold:
					yaw_rate = base_yaw_rate*1.5 #* scale_factor
				elif bbox_center_x > image_center_x + center_threshold:
					yaw_rate = -base_yaw_rate*1.5 #* scale_factor
		elif mode==1:
			# In 'toward' mode, use a fixed yaw rate to align the bounding box center with the image center
			if abs(bbox_center_x - image_center_x) > proximity_threshold:
				if bbox_center_x < image_center_x - center_threshold: # goal is to the left of the image center
					yaw_rate = -toward_yaw_rate  # Turn left to close the gap
				elif bbox_center_x > image_center_x + center_threshold:
					
					yaw_rate = toward_yaw_rate  # Turn left to close the gap
		
	return yaw_rate

def main():
	global obs_img, text_input
	rospy.init_node("LANGUAGE_CONTROLLER", anonymous=False)
	vel_msg = Twist()
	vel_pub = rospy.Publisher(VEL_LANG_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	image_curr_msg = rospy.Subscriber(
		IMAGE_TOPIC, CompressedImage, callback_obs, queue_size=1)
	text_curr_msg = rospy.Subscriber(
		TEXT_TOPIC, String, callback_text, queue_size=1)

	print("waiting for image and text input...")

	while not rospy.is_shutdown():
		# rate.sleep()
		# print("waiting for image and text input...")
		# get image

		# get odometry
		# odom_curr_msg = rospy.Subscriber(
		# 	ODOM_TOPIC, Odometry, callback_odom, queue_size=1)

		# check if text_input and obs_img is well defined
		if text_input_msg.is_valid() and obs_img.is_valid():
		# text_input=input("Enter command: ")
		# if obs_img.is_valid():
			# get text input
			text_input = text_input_msg.get()
			print("recieved command:",text_input)
			image = obs_img.get()
			print(type(image))
			# get velocity command
			vel_msg = Twist()
			w = navigation_policy(
				egocentric_image=image,
				text_input=text_input,
				parsing_function=simple_parser,
				bounding_box_function=predict_boxes,
				proximity_threshold=100, # in pixels
				center_threshold=20, # in pixels
				base_yaw_rate=BASE_YAW_RATE
			)
			vel_msg.angular.z = w
			# publish velocity command
			vel_pub.publish(vel_msg)
			# reset text_input
			# text_input_msg.set(None)
			# reset obs_img
			# obs_img.set(None)

if __name__ == "__main__":
	main()