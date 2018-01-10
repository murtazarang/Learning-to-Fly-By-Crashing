import os
import numpy as np
import cv2 as cv

from alexnet import AlexNet
from caffe_classes import class_names
import tensorflow as tf

import time
from datetime import datetime

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from std_msgs.msg import Empty
from ardrone_autonomy.msg import Navdata
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

from datetime import datetime


imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

class get_image:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/ardrone/image_raw", Image, self.callback)
    self.status = ""
    self.rate = rospy.Rate(1)  # 1Hz

  def callback(self,data):
    try:
      global cv_image
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
      cv.imwrite(str(now) + ".jpg", cv_image)
      #print cv_image.shape
      img_size = cv_image.shape
      left_image = cv_image[0:img_size[0], 0:(img_size[1] / 2)]
      img = cv.resize(left_image, (227, 227))
      # Subtract the ImageNet mean
      mean_img = img - imagenet_mean
      reshape_img = np.reshape(mean_img, (1, 227, 227, 3))
      cv.imshow("Image window", cv_image)
      cv.waitKey(3)
    except CvBridgeError as e:
      print(e)


def initiate_nodes():
    rospy.init_node('FlightControl', anonymous=False)


if __name__ == '__main__':
    initiate_nodes()
    new_frame = get_image()
    rospy.spin()