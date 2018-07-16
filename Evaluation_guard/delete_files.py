# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:25:47 2018

@author: lvsikai
"""
import glob
import os
my_labels = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\5MissFreshSmartShelf_Exp\\dk_labels\\'
my_images = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\4MissFreshSmartShelf_Exp\\JPEGImages\\'
my_images1 = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\4MissFreshSmartShelf_Exp\\JPEGImages\\yuanshi\\'
image_paths = glob.glob(os.path.join(my_images,'*.jpg'))
label_paths = glob.glob(os.path.join(my_images,'*.txt'))
image_paths1 = glob.glob(os.path.join(my_images1,'*.jpg'))
print image_paths1
'''
for lblpath in label_paths:
    lblname = lblpath.split('\\')[-1].strip()[0:-4]
    for imgpath in image_paths:
        imgname = imgpath.split('\\')[-1].strip()[0:-4]
        if(lblname==imgname):
            continue
'''
count=0
#delete files

for imgpath1 in image_paths1:
    name1 = imgpath1.split('\\')[-1].strip()[0:-4]
    for imgpath in image_paths:
        name = imgpath.split('\\')[-1].strip()[0:-4]
        if(name1==name):
            print('name:',name)
            count+=1
            os.remove(imgpath)
