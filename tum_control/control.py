#!/usr/bin/env python
''' add following dependencies
sensor_msgs
opencv2
cv_bridge
rospy
std_msgs
'''

import rospy
import roslib
import sys

from std_msgs.msg import String
from std_msgs.msg import Empty

from ardrone_autonomy.msg import Navdata
from geometry_msgs.msg import Twist

from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf

import numpy as np



CONNECTION_PERIOD = 5
GUI_UPDATE = 0.2

class FlightControl():
    def __init__(self):
        self.status = ""
        rospy.init_node('FlightControl', anonymous=False)
        self.rate = rospy.Rate(10) #10Hz
        self.pubTakeOff = rospy.Publisher("ardrone/takeoff", Empty, queue_size=20)
        self.pubLand = rospy.Publisher("ardrone/land", Empty, queue_size=20)
        self.pubControlVel = rospy.Publisher("cmd_vel", Twist, queue_size=20)
        self.ControlVel = Twist()
        
    def ControlTakeOff(self):
        self.pubTakeOff.publish(Empty())
        self.rate.sleep()
    
    def ControlLand(self):
        self.pubLand.publish(Empty)
        
    def SendControlVel(self, lin_x, lin_y, lin_z, ang_x, ang_y, ang_z):
        self.ControlVel.linear.x = lin_x
        self.ControlVel.linear.y = lin_y
        self.ControlVel.linear.z = lin_z
        self.ControlVel.angular.x = ang_x
        self.ControlVel.angular.y = ang_y
        self.ControlVel.angular.z = ang_z
        self.pubControlVel.publish(self.ControlVel)
        self.rate.sleep()

class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("image_topic",Image,self.callback)
    #rospy.init_node('image_converter', anonymous=True)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      print cv_image.shape
    except CvBridgeError as e:
      print(e)

    cv.imshow("Image window", cv_image)
    cv.waitKey(3)



'''        
class VideoFeed():
    def __init__(self):
        self.subNavData = rospy.Subscriber('/ardrone/navdata', Navdata,self.ReceiveNavData)
        self.subVideo = rospy.Subscriber('/ardrone/image_raw', Image, self.ReceiveImage)
        
        self.image = None
        self.imageLock = Lock()
        cv.NamedWindow("windowimage", cv.CV_Window_Autosize)
        
        self.statusMessage = ''
        
        self.CommunicationCheck = False
        self.connected = False
        
        
        self.image_sub = rospy.Subscriber
        self.CommunicationCheck = rospy.Timer(rospy.Duration(CONNECTION_PERIOD), self.ConnectionCallback)
        self.redrawTimer = rospy.Timer(rospy.Duration(GUI_UDATE), self.RedrawCallback)
        
        
    def ConnectionCallback(self,event):
        self.connected = self.CommunicationCheck
        self.CommunicationCheck - False
        
        
    def RedrawCallback(self,event):
        if self.image is not None:
            self.imageLock.acquire()
            try:
                image_cv = ToOpenCV(self.image)
            finally:
                self.imageLock.release()
            
            cv.imshow("windowimage", image_cv)
            cv.WaitKey(3)
            
    def ReceiveImage(self,data):
        self.CommunicationCheck = True
        
        self.imagelock.acquire()
        try:
            self.image = data
        finally:
            self.imageLock.release()
            
    def ReceiveNavdata(self,navdata):
        self.CommunicationCheck = True
   '''

"""

"""
        
if __name__ == '__main__': 
       
    uav = FlightControl()
    ic = image_converter()
    
    try: 
        i = 0
               
        while not rospy.is_shutdown():
            uav.ControlTakeOff()
            if i <= 40 :
                uav.SendControlVel(0,0,1,0,0,0)
                i+=1
            elif i<=70 :
                uav.SendControlVel(1,0,0,0,0,0)
                i+=1
            else:
                uav.SendControlVel(0,0,0,0,0,0)
         
    except rospy.ROSInterruptException:
        pass
