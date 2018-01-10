#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:22:16 2017

@author: tagomago
"""

import csv
from shutil import copy2
import os.path

folderpath = "/media/tagomago/Orchid/publish_data/"
impact_filename = 'hit_time.csv'
image_seq_filename = 'img_seq_time.csv'
dest_path = "/home/tagomago/crash_images_data/"

count = 1

while count <= 11001:
    impact_time = 0
    current_test_folder = folderpath + '{0:05}'.format(count)
    if os.path.exists(os.path.join(current_test_folder, impact_filename)):
        im_time_filepath = os.path.join(current_test_folder, impact_filename)
        
        with open(im_time_filepath, 'rb') as f:
            impact_time_file = csv.reader(f)
            mycsv = list(impact_time_file)
            impact_time = mycsv[1][0]
    
    
        image_seq_time_filepath = os.path.join(current_test_folder, image_seq_filename)
        with open(image_seq_time_filepath, 'rb') as f:
            image_seq_file =csv.reader(f)
            mycsv = list(image_seq_file)
            i = 1
            while i <= (len(mycsv)-1):
                actual_time_diff = float(impact_time) - float(mycsv[i][1])
                
                if float(impact_time) < 1.0:
                    if actual_time_diff <= 
                    
                    if actual_time_diff <= des_impact_time_delta:
                        crash_image = mycsv[i][0]
                        copy2(os.path.join(current_test_folder, crash_image), os.path.join(dest_path, crash_image))
                i+=1
    count +=1