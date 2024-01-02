"""
take in microphone input on keypress
query gcloud to transcribe to text
parse text into object and mode
"""
import os
import sys
import rospy

from std_msgs.msg import String

TEXT_TOPIC = "/text_input"

def main():
    # Your code here
    rospy.init_node("TEXT_INPUT", anonymous=False)
    text_pub = rospy.Publisher(TEXT_TOPIC, String, queue_size=1)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        text = input("Enter command: ")
        text_pub.publish(text)
        rate.sleep()

if __name__ == "__main__":
    main()