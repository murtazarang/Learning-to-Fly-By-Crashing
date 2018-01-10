import sys
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2 as cv

from alexnet import AlexNet
from caffe_classes import class_names
import tensorflow as tf

import time
from datetime import datetime
"""
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import String
from std_msgs.msg import Empty
from ardrone_autonomy.msg import Navdata
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
"""
import csv

imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
ang_vel_const = 0.02
ex_acc = 0.75

#cv_image = np.zeros((360,640,3), dtype=np.float32)

filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("/home/tagomago/test_images/*.jpg"))

image_reader = tf.WholeFileReader()

_, image_file = image_reader.read(filename_queue)

image = tf.image.decode_jpeg(image_file)

img_resized = tf.image.resize_images(image, [227, 227])

img_centered = tf.subtract(img_resized, imagenet_mean)

# input layer image dimensions
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
# dropout rate
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, 2, [])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('/home/tagomago/finetune_alexnet_3/checkpoints/model_epoch57.ckpt.meta')
    saver.restore(sess, "/home/tagomago/finetune_alexnet_3/checkpoints/model_epoch57.ckpt")








image_path = '/home/tagomago/test_images'

# input layer image dimensions
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
# dropout rate
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, 2, [])
score = model.fc8
softmax = tf.nn.softmax(score)

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



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('/home/tagomago/finetune_alexnet_3/checkpoints/model_epoch57.ckpt.meta')
    saver.restore(sess, "/home/tagomago/finetune_alexnet_3/checkpoints/model_epoch57.ckpt")

    onlyfiles = [f for f in listdir(image_path) if isfile(join(image_path, f))]

    #img_paths = convert_to_tensor(onlyfiles, dtype=dtypes.string)

    cv_image = np.empty(len(onlyfiles), dtype=object)



    for n in range(0, len(onlyfiles)):
        #img_string = tf.read_file(join(image_path, onlyfiles[n])
        #img_decoded = tf.image.decode_png(img_string, channels=3)
        #img_resized = tf.image.resize_images(img_decoded, [227, 227])
        #img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        #cv_image[n] = img_centered[:, :, ::-1]
        cv_image[n] = cv.imread(join(image_path, onlyfiles[n]))
        print onlyfiles[n]

    for n in range(0, len(onlyfiles)):
        f_image = process_image(cv_image[n], 0)
        l_image = process_image(cv_image[n], 1)
        r_image = process_image(cv_image[n], 2)
        f_m_img = resize_image(f_image)
        l_m_img = resize_image(l_image)
        r_m_img = resize_image(r_image)
        s_prob = calc_prob(f_m_img)
        l_prob = calc_prob(l_m_img)
        r_prob = calc_prob(r_m_img)
        with open('text_values.csv', 'ab') as file:
            file.write('Image'+str(n)+',')
            file.write(str(s_prob)+ ',')
            file.write(str(l_prob)+ ',')
            file.write(str(r_prob)+ ',')
            file.write('\n')

