# -*- coding: utf-8 -*-
"""
Created on Wed May 02 17:40:56 2018

@author: lvsikai
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
import json

import pdb

dataset_dir = "./"

# raw images(*.jpg) dir
raw_images_dir = os.path.join(dataset_dir,'JPEGImages')

# pascal-voc labels(*.xml) dir
voc_labels_dir = os.path.join(dataset_dir,"Annotation\\JPEGImages")

# pascal-voc labels(*.txt) dir
in_file_dir = os.path.join(dataset_dir,"dk_labels1")
out_file_dir = os.path.join(dataset_dir,"dk_labels")

# imageset dir(train.txt/val.txt)
imageset_dir = os.path.join(dataset_dir,'ImageSets','Main')

dataset_file = os.path.join(imageset_dir, 'train.txt')
f = open(dataset_file)
file_names = f.readlines()
f.close()

file_names = [x.strip() for x in file_names]

for name in file_names:
    inf = open(os.path.join(in_file_dir,'%s.txt'%name),'r')
    outf = open(os.path.join(out_file_dir,'%s.txt'%name),'w')
    object_list = inf.readlines()
    object_list = [x.strip().split(' ') for x in object_list]#
    inf.close()
    for object in object_list:
        if object[0] == '0' or object[0] == '5' or object[0] =='14' or object[0] =='15' or object[0] =='17' or object[0] =='18' or object[0] =='19' or object[0] =='21' or object[0] =='22' or object[0] =='23' :
            outf.write('0 '+object[1]+' '+object[2]+' '+object[3]+' '+object[4]+'\n')
        elif object[0] == '1' or object[0] == '2' or object[0] =='6' or object[0] =='7' or object[0] =='9' or object[0] =='13' or object[0] =='16':
            outf.write('1 '+object[1]+' '+object[2]+' '+object[3]+' '+object[4]+'\n')
        elif object[0] == '3' or object[0] == '4':
            outf.write('2 '+object[1]+' '+object[2]+' '+object[3]+' '+object[4]+'\n')
        elif (object[0] == '8'):
            outf.write('3 '+object[1]+' '+object[2]+' '+object[3]+' '+object[4]+'\n')
        else:
            outf.write('4 '+object[1]+' '+object[2]+' '+object[3]+' '+object[4]+'\n')
    outf.close()