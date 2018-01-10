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


imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
ang_vel_const = 0.02
ex_acc = 0.4

#cv_image = np.zeros((360,640,3), dtype=np.float32)

# input layer image dimensions
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
# dropout rate
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, 2, [])
score = model.fc8
softmax = tf.nn.softmax(score)

bridge = CvBridge()

class FlightControl():
    def __init__(self):
        self.status = ""
        self.rate = rospy.Rate(1)  # 1Hz
        self.pubTakeOff = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=1)
        self.pubLand = rospy.Publisher("/ardrone/land", Empty, queue_size=1)
        self.pubControlVel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
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

def initiate_nodes():
    rospy.init_node('FlightControl', anonymous=False)

class get_image:

  def __init__(self):

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/ardrone/image_raw", Image, self.callback)

  def callback(self,data):
    try:
      global cv_image
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      #print cv_image.shape
      cv.imshow("Image window", cv_image)
      cv.waitKey(3)
    except CvBridgeError as e:
      print(e)
"""
def single_image():
    cv_image = rospy.wait_for_message("/ardrone/front/image_raw", Image)
    camera_image = bridge.imgmsg_to_cv2(cv_image, "bgr8")
    #cv.imwrite("image" + str(i) +".jpeg", camera_image)
    return camera_image
"""
"""
def get_image():
    try:
        cv_image = rospy.Subscriber('/ardrone/front/image_raw', Image, queue_size=1)
        camera_image = bridge.imgmsg_to_cv2(cv_image, "bgr8")
        return camera_image
    except CvBridgeError, e:
        print(e)

class image_converter:

    def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/ardrone/front/image_raw', Image, self.callback, queue_size=1)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
"""
def process_image(camera_image, i):
    if i == 0:
        front_image = camera_image  # straight image
        return front_image
    elif i == 1:
        img_size = camera_image.shape
        print img_size
        left_image = camera_image[0:img_size[0], 0:(img_size[1] / 2)]  # left image
        return left_image
    elif i == 2:
        img_size = camera_image.shape
        print img_size
        right_image = (camera_image[0:img_size[0], (img_size[1] / 2):img_size[1]])  # right image
        return right_image

def resize_image(image):
        # resize to (227x227)
        img = cv.resize(image, (227, 227))
        # Subtract the ImageNet mean
        mean_img = img - imagenet_mean
        reshape_img = np.reshape(mean_img, (1, 227, 227, 3))
        #cv.imwrite("meanimage" + str(i) +".jpeg", reshape_image)
        return reshape_img

def calc_prob(mean_img):
    probs = sess.run(softmax, feed_dict={x: mean_img, keep_prob: 1})
    print probs
    return probs


if __name__ == '__main__':
    initiate_nodes()
    uav = FlightControl()
    new_frame = get_image()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('/home/tagomago/finetune_alexnet_2/checkpoints/model_epoch10.ckpt.meta')
        saver.restore(sess, "/home/tagomago/finetune_alexnet_2/checkpoints/model_epoch10.ckpt")
        i = 0
        count = 0
        while True:
            try:
                while not rospy.is_shutdown():
                    uav.ControlTakeOff()
                    #camera_image = single_image()
                    #cv.imshow("initial state1", camera_image)
                    f_image = process_image(cv_image, 0)
                    l_image = process_image(cv_image, 1)
                    r_image = process_image(cv_image, 2)
                    f_m_img = resize_image(f_image)
                    #cv.imwrite("/home/tagomago/test_run/first_loop/f" + str(count) + ".png", f_m_img)
                    #cv.imshow("initial front state", f_m_img)
                    #cv.waitKey()
                    l_m_img = resize_image(l_image)
                    #cv.imwrite("/home/tagomago/test_run/first_loop/l" + str(count) + ".png", l_m_img)
                    #cv.imshow("initial left state", l_m_img)
                    #cv.waitKey(3)
                    r_m_img = resize_image(r_image)
                    #cv.imwrite("/home/tagomago/test_run/first_loop/r" + str(count) + ".png", r_m_img)
                    #cv.imshow("initial right state", r_m_img)
                    #cv.waitKey()
                    count += 1
                    s_prob = calc_prob(f_m_img)
                    l_prob = calc_prob(l_m_img)
                    r_prob = calc_prob(r_m_img)

                    if s_prob[0,0] > ex_acc:
                        ang_vel = -0.1 * ang_vel_const * (r_prob[0,0]-l_prob[0,0])
                        print "Straight Pred " + str(s_prob[0, 0])
                        print "Left Pred " + str(l_prob[0, 0])
                        print "Right Pred " + str(r_prob[0, 0])
                        uav.SendControlVel(0.5,0,0,0,0,0)
                        time.sleep(0.5)
                        uav.SendControlVel(0, 0, 0, 0, 0, 0)
                    else:
                        uav.SendControlVel(0,0,0,0,0,0)
                        if r_prob[0,0] > l_prob[0,0]:
                            while s_prob[0,0] < ex_acc:
                                #camera_image = single_image()
                                #cv.imshow("initial state", camera_image)
                                #cv.waitKey()
                                f_image = process_image(cv_image, 0)
                                f_m_img = resize_image(f_image)
                                #cv.imwrite("/home/tagomago/test_run/first_loop/f" + str(count) + ".png", f_m_img)
                                count += 1
                                #cv.imshow("initial front state", f_m_img)
                                #cv.waitKey()
                                s_prob = calc_prob(f_m_img)
                                print "Prob R>L, Straight pred " + str(s_prob[0, 0]) + " rotate to right"
                                print "Left Pred " + str(l_prob[0, 0])
                                print "Right Pred " + str(r_prob[0, 0])
                                uav.SendControlVel(0,0,0,0,0,-0.5)
                                time.sleep(0.5)
                                uav.SendControlVel(0, 0, 0, 0, 0, 0)
                        else:
                            while s_prob[0,0] < ex_acc:
                                #camera_image = single_image()
                                #cv.imshow("initial state", camera_image)
                                #cv.waitKey()
                                f_image = process_image(cv_image, 0)
                                f_m_img = resize_image(f_image)
                                #cv.imwrite("/home/tagomago/test_run/first_loop/f" + str(count) + ".png", f_m_img)
                                count += 1
                                #cv.imshow("initial front state", f_m_img)
                                #cv.waitKey(500)
                                s_prob = calc_prob(f_m_img)
                                print "Prob L>R, Straight pred " + str(s_prob[0,0]) + " rotate left"
                                print "Left Pred " + str(l_prob[0, 0])
                                print "Right Pred " + str(r_prob[0, 0])
                                uav.SendControlVel(0, 0, 0, 0, 0, 0.5)
                                time.sleep(0.5)
                                uav.SendControlVel(0, 0, 0, 0, 0, 0)

            except rospy.ROSInterruptException:
                pass