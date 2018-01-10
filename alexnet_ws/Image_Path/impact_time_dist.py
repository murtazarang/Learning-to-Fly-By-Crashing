#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 06:08:59 2017

@author: tagomago
"""

import csv
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import scipy.stats as stats
import numpy as np

folderpath = "/media/tagomago/Orchid/publish_data/"
impact_filename = 'hit_time.csv'
image_seq_filename = 'img_seq_time.csv'
dest_path = "/home/tagomago/crash_images_data/"

count = 1

impact_time = []

while count <= 11001:
    
    current_test_folder = folderpath + '{0:05}'.format(count)
    if os.path.exists(os.path.join(current_test_folder, impact_filename)):
        im_time_filepath = os.path.join(current_test_folder, impact_filename)
        with open(im_time_filepath, 'rb') as f:
            impact_time_file = csv.reader(f)
            mycsv = list(impact_time_file)
            impact_time.append(float(mycsv[1][0]))
    count+=1


t = [float(i) for i in impact_time]

t1 = sorted(t)
print t1

'''
t_mean = np.mean(t)
t_std = np.std(t)
print t_mean
pdf = stats.norm.pdf(t,t_mean, t_std)
plt.plot(t, pdf)
'''