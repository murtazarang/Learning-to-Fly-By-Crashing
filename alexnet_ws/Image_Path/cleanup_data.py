#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:45:32 2017

@author: tagomago
"""

import csv
from shutil import copy2
from shutil import copytree
import os, sys
import fnmatch

folderpath = "/home/tagomago/publish_data"
impact_filename = 'hit_time.csv'
image_seq_filename = 'img_seq_time.csv'
outlier_path = "/home/tagomago/outliers/"

count = 3378

while count <= 3380:
    impact_time = 0
    image_list = []
    file_list = []
    current_outlier_path = outlier_path + '{0:05}'.format(count)
    #print os.path.exists(current_test_folder)
    if os.path.exists(current_outlier_path):
        test_folder_name = '{0:05}'.format(count)
        path = os.path.join(folderpath, test_folder_name)
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, '*.jpeg'):
                image_list.append(file)
        image_list = sorted(image_list)
        #print image_list
        image_seq_time_filepath = os.path.join(path, image_seq_filename)
        start_row = 1
        with open(image_seq_time_filepath, 'rb') as f:
            image_seq_file = csv.reader(f)
            mycsv = list(image_seq_file)
                            
        with open(image_seq_time_filepath, 'wb') as f:
            image_seq_writer = csv.writer(f)
            for row in mycsv:
                if row == ["#img", "time"]:
                           image_seq_writer.writerow(row)
                
                if row[0] in image_list:
                    image_seq_writer.writerow(row)
            
            
            
            #update start time
        with open(image_seq_time_filepath, 'rb') as f:
            start_time = csv.reader(f)
            mycsv_start_time = list(start_time)
            #print mycsv_start_time
            new_start_time = mycsv_start_time[1][1]
                
        with open(image_seq_time_filepath, 'wb') as f:
            update_time_writer = csv.writer(f)
            for row in mycsv_start_time:
                if row == ["#img", "time"]:
                           update_time_writer.writerow(row)
                else:
                           row[1] = str(float(row[1]) - float(new_start_time))
                           update_time_writer.writerow(row)
        
        impact_time_path = os.path.join(path, impact_filename)
        
        with open(impact_time_path, 'rb') as f:
            old_impact_file = csv.reader(f)
            mycsv_old_impact = list(old_impact_file)
        
        with open(impact_time_path, 'wb') as f:
            update_impact_writer = csv.writer(f)
            for row in mycsv_old_impact:
                if row[0] == "#time":
                    update_impact_writer.writerow(row)
                else:
                    row[0] = str(float(row[0]) - float(new_start_time))
                    update_impact_writer.writerow(row)
                
        
    count += 1