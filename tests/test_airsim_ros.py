#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage


import airsim

class AirSimMultirotorControl:
    def __init__(self):
        # Initialize the AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)

        print("AirSim connection established and API control enabled.")

    def takeoff(self):
        # Take off the drone
        print("Taking off...")
        self.client.takeoffAsync().join()

    def move_forward(self, duration, speed):
        # Move the drone forward
        print(f"Moving forward for {duration} seconds at {speed} m/s.")
        self.client.moveByVelocityAsync(vx=speed, vy=0, vz=0, duration=duration).join()

    # def take_picture(self):
    #     # Take a picture
    #     print("Taking a picture...")
    #     responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    #     return responses[0].image_data_uint8

    def land(self):
        # Land the drone
        print("Landing...")
        self.client.landAsync().join()
        self.client.enableApiControl(False)
        print("Landed and API control disabled.")


# Global variables
image_count = 0
airsim_control = AirSimMultirotorControl()
MAX_COUNT=10
FREQ=10
rospy.init_node('image_listener', anonymous=True)
rate = rospy.Rate(1)

def image_callback(msg):
    global image_count
    global rate
    


    image_count += 1

    if image_count%FREQ==0 and image_count//FREQ < MAX_COUNT:
        # Save the image
        idx=image_count//FREQ
        image_filename = f"received_image_{idx}.jpg"
        with open(image_filename, "wb") as img_file:
            img_file.write(msg.data)
        print("Saved an image!")
        # Take another picture after moving forward
        # airsim_control.move_forward(duration=1, speed=10)  # Adjust duration and speed as needed
        # picture = airsim_control.take_picture()
        # Do something with the picture if needed
        
    # if you signal_shutdown here, nothing to catch the image_callback outputs
    # else:
    #     print(f"Saved {MAX_COUNT} images, shutting down.")
        # airsim_control.land()
        # rospy.signal_shutdown("Received 5 images")
    # rate.sleep()

def main():
    
    
    # AirSim operations
    airsim_control.takeoff()
    rospy.Subscriber("/airsim_node/SimpleFlight/front_center/Scene/compressed", CompressedImage, image_callback)
    airsim_control.move_forward(duration=10, speed=20)  # Adjust duration and speed as needed

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    finally:
        airsim_control.land()
