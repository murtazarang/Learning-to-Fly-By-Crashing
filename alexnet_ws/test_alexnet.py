import os
import numpy as np
import cv2 as cv

from alexnet import AlexNet
from caffe_classes import class_names
import tensorflow as tf

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from std_msgs.msg import Empty
from ardrone_autonomy.msg import Navdata
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError


imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
ang_vel_const = 0.5
ex_acc = 0.9


"""
current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'images')

img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]
"""

# input layer image dimensions
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
# dropout rate
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, 1000, [])
score = model.fc8
softmax = tf.nn.softmax(score)

class FlightControl():
    def __init__(self):
        self.status = ""
        rospy.init_node('FlightControl', anonymous=False)
        self.rate = rospy.Rate(10)  # 10Hz
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
        self.image_sub = rospy.Subscriber("image_topic", Image, self.callback)

        #rospy.init_node('image_converter', anonymous=False)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            return cv_image
        except CvBridgeError as e:
            print(e)

        cv.imshow("Image window", cv_image)
        cv.waitKey(3)

class RosTensorFlow():
    def __init__(self):

        self.status = ""
        rospy.init_node('FlightControl', anonymous=False)
        self.rate = rospy.Rate(10)  # 10Hz
        self.pubTakeOff = rospy.Publisher("ardrone/takeoff", Empty, queue_size=20)
        self.pubLand = rospy.Publisher("ardrone/land", Empty, queue_size=20)
        self.pubControlVel = rospy.Publisher("cmd_vel", Twist, queue_size=20)
        self.ControlVel = Twist()

        self.sess = tf.Session()
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('/ardrone/image_raw', Image, self.callback, queue_size=1)
        #self._pub = rospy.Publisher('result', String, queue_size=1)
        #self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        #self.use_top_k = rospy.get_param('~use_top_k', 5)


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

    def callback(self, data):
        cv_image = self._cv_bridge.imgmsg_to_cv2(data, "bgr8")
        # copy from
        # classify_image.py
        images = []
        coll_prob = []
        img_size = cv_image.shape
        images.append(cv_image)  # straight image
        images.append(cv_image[0:img_size[0], 0:(img_size[1] / 2)])  # left image
        images.append(cv_image[0:img_size[0], (img_size[1] / 2):img_size[1]])  # right image

    def main(self):
        rospy.spin()
        saver = tf.train.import_meta_graph('/home/tagomago/finetune_alexnet/checkpoints/model_epoch10.ckpt.meta')
        saver.restore(self.sess, "/home/tagomago/finetune_alexnet/checkpoints/model_epoch10.ckpt")



if __name__ == '__main__':
    # Now let's run it
    uav = FlightControl()
    ic = image_converter()

    with tf.Session() as sess:
        #saver = tf.train.import_meta_graph('/home/tagomago/finetune_alexnet/checkpoints/model_epoch10.ckpt.meta')
        #saver.restore(sess, "/home/tagomago/finetune_alexnet/checkpoints/model_epoch10.ckpt")
        # model.load_initial_weights(sess)
        # camera_image = ic.callback()
        # create images to be parsed
    while True:
        try:

                while not rospy.is_shutdown():
                    uav.ControlTakeOff()
                    camera_image = ic.callback()
                    # create images to be parsed
                    coll_prob = []
                    pred_class = []
                    images = []
                    img_size = camera_image.shape
                    images.append(camera_image)  # straight image
                    images.append(camera_image[0:img_size[0], 0:(img_size[1] / 2)])  # left image
                    images.append(camera_image[0:img_size[0], (img_size[1] / 2):img_size[1]])  # right image

                    for i, image in enumerate(images):
                        # Convert image to float32 and resize to (227x227)

                        img = cv.resize(image, (227, 227))

                        # Subtract the ImageNet mean
                        mean_img = img - imagenet_mean

                        # Reshape as needed to feed into model
                        # img = img.reshape((1, 227, 227, 3))

                        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})

                        # class names with highest probability
                        coll_prob.append(np.argmax(probs))
                        pred_class.append(class_names[np.argmax(probs)])

                    if probs[0]>ex_acc:
                        ang_vel = ang_vel_const * (probs(2)-probs(1))
                        uav.SendControlVel(1, 0, 0, 0, 0, ang_vel)
                    else:
                        uav.SendControlVel(0, 0, 0, 0, 0, 0)
                        if probs[2]>probs[1]:
                            while probs[0]<ex_acc:
                                camera_image = ic.callback()
                                # create images to be parsed
                                coll_prob = []
                                pred_class = []
                                images = []
                                img_size = camera_image.shape
                                images.append(camera_image)  # straight image
                                #images.append(camera_image[0:img_size[0], 0:(img_size[1] / 2)])  # left image
                                #images.append(camera_image[0:img_size[0], (img_size[1] / 2):img_size[1]])  # right image

                                for i, image in enumerate(images):
                                    # Convert image to float32 and resize to (227x227)

                                    img = cv.resize(image, (227, 227))

                                    # Subtract the ImageNet mean
                                    mean_img = img - imagenet_mean

                                    # Reshape as needed to feed into model
                                    # img = img.reshape((1, 227, 227, 3))

                                    probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})

                                    # class names with highest probability
                                    coll_prob.append(np.argmax(probs))
                                    pred_class.append(class_names[np.argmax(probs)])
                                    uav.SendControlVel(0, 0, 0, 0, 0, -0.5)
                        else:
                            while probs[0] < ex_acc:
                                camera_image = ic.callback()
                                # create images to be parsed
                                coll_prob = []
                                pred_class = []
                                images = []
                                img_size = camera_image.shape
                                images.append(camera_image)  # straight image
                                #images.append(camera_image[0:img_size[0], 0:(img_size[1] / 2)])  # left image
                                #images.append(camera_image[0:img_size[0], (img_size[1] / 2):img_size[1]])  # right image

                                for i, image in enumerate(images):
                                    # Convert image to float32 and resize to (227x227)

                                    img = cv.resize(image, (227, 227))

                                    # Subtract the ImageNet mean
                                    mean_img = img - imagenet_mean

                                    # Reshape as needed to feed into model
                                    # img = img.reshape((1, 227, 227, 3))

                                    probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})

                                    # class names with highest probability
                                    coll_prob.append(np.argmax(probs))
                                    pred_class.append(class_names[np.argmax(probs)])
                                    uav.SendControlVel(0, 0, 0, 0, 0, 0.5)

        except rospy.ROSInterruptException:
            pass


    print("Original files", img_files)