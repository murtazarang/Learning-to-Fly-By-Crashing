#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 03:25:48 2017

@author: tagomago
"""
import os
import random
import csv


folderpath = "/home/tagomago/publish_data/"
impact_filename = 'hit_time.csv'
image_seq_filename = 'img_seq_time.csv'
dest_path = "/home/tagomago/crash_images_data/"
text_filepath = "/home/tagomago/tum_simulator_ws/src/alexnet_ws/"

count = 1

small_test_faraway = 0.3
small_test_coll = 0.3
large_test_coll = 0.2
large_test_faraway = 0.3
vlarge_test_coll = 0.15
vlarge_test_faraway = 0.20

train_text_path = os.path.join(text_filepath, "train.txt")
val_text_path = os.path.join(text_filepath, "val.txt")

far_text_path = os.path.join(text_filepath, "path_far.txt")
coll_text_path = os.path.join(text_filepath, "path_coll.txt")

open(far_text_path, "wb").close
open(coll_text_path, "wb").close
open(train_text_path, "wb").close
open(val_text_path, "wb").close

while count <= 11033:
    test_folder = '{0:05}'.format(count)
    current_folder = os.path.join(folderpath, test_folder) 
    
    if os.path.exists(current_folder):
        lists = os.listdir(current_folder)
        lists = sorted(lists)
        num_images = len(lists) - 3
        
        if os.path.exists(os.path.join(current_folder, impact_filename)):
            im_time_filepath = os.path.join(current_folder, impact_filename)
            with open(im_time_filepath, 'rb') as t:
                impact_time_file = csv.reader(t)
                mycsv_impact_time = list(impact_time_file)
                impact_time = float(mycsv_impact_time[1][0])
        
        if num_images <= 50 and num_images != 0:        #define first 30% as far away, and last 30% as collision
            
            small_faraway_list_size = int(small_test_faraway * num_images)
            small_coll_list_size = num_images - int(small_test_coll * num_images)
            
            with open(far_text_path,"ab") as f:
                i = 0
                
                while i < small_faraway_list_size:
                    current_image = lists[i]
                    print_path_class = os.path.join(current_folder, current_image) + ' 0'
                    f.write("{}\n".format(print_path_class))
                    i += 1
            
            with open(coll_text_path,"ab") as f:   
                j = num_images - 1
                while j > small_coll_list_size:
                    current_image = lists[j]
                    print_path_class = str(os.path.join(current_folder, current_image)) + ' 1'
                    f.write("{}\n".format(print_path_class))
                    j -= 1
                    
        if num_images > 50 and num_images <=150:        #define first 30% as far away, and last 30% as collision
            
            large_faraway_list_size = int(large_test_faraway * num_images)
            large_coll_list_size = num_images - int(large_test_coll * num_images)
            
            text_path = os.path.join(text_filepath, "path_far.txt")
            with open(far_text_path,"ab") as f:
                i = 0
                
                while i < large_faraway_list_size:
                    current_image = lists[i]
                    print_path_class = os.path.join(current_folder, current_image) + ' 0'
                    f.write("{}\n".format(print_path_class))
                    i += 1
           
            
            with open(coll_text_path,"ab") as f:       
                j = num_images - 1
                while j > large_coll_list_size:
                    current_image = lists[j]
                    print_path_class = os.path.join(current_folder, current_image) + ' 1'
                    f.write("{}\n".format(print_path_class))
                    j -= 1

        if num_images > 100 and num_images <= 400:        #define first 20% as far away, and last 15% as collision
            
            vlarge_faraway_list_size = int(vlarge_test_faraway * num_images)
            vlarge_coll_list_size = num_images - int(vlarge_test_coll * num_images)
            
            with open(far_text_path,"ab") as f:
                i = 0
                
                while i < vlarge_faraway_list_size:
                    current_image = lists[i]
                    print_path_class = os.path.join(current_folder, current_image) + ' 0'
                    f.write("{}\n".format(print_path_class))
                    i += 1
            
            with open(coll_text_path,"ab") as f:      
                j = num_images - 1
                while j > vlarge_coll_list_size:
                    current_image = lists[j]
                    print_path_class = os.path.join(current_folder, current_image) + ' 1'
                    f.write("{}\n".format(print_path_class))
                    j -= 1
    count += 1
                    
                    
def text_file_len(fname):
    with open(fname) as f:
        for k,l in enumerate(f):
            pass
    return k+1

coll_text_path = os.path.join(text_filepath, "path_coll.txt")
far_text_path = os.path.join(text_filepath, "path_far.txt")
faraway_set_size = text_file_len(far_text_path)
coll_set_size = text_file_len("path_coll.txt")

#generate test.txt
test_percent = 0.8
test_far_set_size = int(test_percent * faraway_set_size)
test_coll_set_size = int(test_percent * coll_set_size)

with open(far_text_path, 'r+') as f: #open in read / write mode
    f.readline() #read the first line and throw it out
    data = f.read() #read the rest
    f.seek(0) #set the cursor to the top of the file
    f.write(data) #write the data back
    f.truncate() #set the file size to the current size

with open(coll_text_path, 'r+') as f: #open in read / write mode
    f.readline() #read the first line and throw it out
    data = f.read() #read the rest
    f.seek(0) #set the cursor to the top of the file
    f.write(data) #write the data back
    f.truncate() #set the file size to the current size

with open(far_text_path, 'rb') as f:
    faraway_data = f.read().split('\n')

with open(coll_text_path, 'rb') as f:
    coll_data = f.read().split('\n')


random.shuffle(faraway_data)
random.shuffle(coll_data)

far_train_data = faraway_data[:test_far_set_size]
far_val_data = faraway_data[test_far_set_size:]

coll_train_data = coll_data[:test_coll_set_size]
coll_val_data = coll_data[test_coll_set_size:]



with open(train_text_path, 'wb') as f:
    for line in far_train_data:
        f.write("\n%s" % line)
    for line in coll_train_data:
        f.write("\n%s" % line)    

with open(val_text_path, 'wb') as f:
    for line in coll_val_data:
        f.write("\n%s" % line)  
    for line in far_val_data:
        f.write("\n%s" % line) 

print text_file_len(val_text_path)
print text_file_len(train_text_path)