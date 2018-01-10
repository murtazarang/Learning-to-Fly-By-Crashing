#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 07:06:29 2017

@author: tagomago
"""

import csv
from shutil import copy2
from shutil import copytree
import os.path

folderpath = "/media/tagomago/Orchid/publish_data/"
impact_filename = 'hit_time.csv'
image_seq_filename = 'img_seq_time.csv'
dest_path = "/home/tagomago/outliers/"

count = 1

while count <= 11001:
    impact_time = 0
    current_test_folder = folderpath + '{0:05}'.format(count)
    #print os.path.exists(current_test_folder)
    if os.path.exists(current_test_folder):
        
        im_time_filepath = os.path.join(current_test_folder, impact_filename)
       
        with open(im_time_filepath, 'rb') as f:
        
            impact_time_file = csv.reader(f)
            mycsv = list(impact_time_file)
            impact_time = mycsv[1][0]
            
            if float(impact_time) > 20 and float(impact_time) < 40:
                for item in os.listdir(current_test_folder):
            
                    dest_path_current = dest_path + '{0:05}'.format(count)
                   
                    if not os.path.exists(dest_path_current):
                        os.mkdir(dest_path_current)
                        if os.path.isdir(os.path.join(current_test_folder, item)):
                            copytree(os.path.join(current_test_folder, item),os.path.join(dest_path_current, item))
                        else:
                            copy2(os.path.join(current_test_folder, item),os.path.join(dest_path_current, item))
    count += 1