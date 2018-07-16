#coding: utf-8

import os,sys
from darknet import *
#import cv2
import pdb
#from utils import *
import multiprocessing as mp

from conf import *
import time
import datetime



import cv2
from conf import *
#from Get_Washed_Video_Frame import get_key_frame_and_info
from image_feature_extractor import Feature_Extractor
from Make_Video_Feature_Sets import Video_Feature_Sets
from Search_Video_by_Image import content_base_video_retrieval
#from util import sec_to_hms

"""
API for MissFresh Project
Run object detection on images, and get results
"""

# client provided internal goods_id 
classes_id = ['image-p-qcshnt-268ml','image-p-xxwqaywsn-2h',
              'image-p-xxwxjnn-4h','image-p-kkkl-330ml',
              'image-p-hnwssgnyl-250ml*4','image-p-sdlwlcwt-4p',
              'image-p-mncnn250ml','image-p-wtnmc-250ml',
              'image-p-lfyswnc-280ml','image-p-hbahtkkwmyr-250ml',
              'gcht','ynhlg',
              'celxl-4','image-p-hyd-mnsnn-4h',
              'image-p-nfnfccz-300ml','image-p-yydhgc235ml',
              'image-p-mqtzyl-1h','image-p-nfsqcpyzlc-500ml',
              'image-p-blkqsyw-1b','image-p-szsj-350ml',
              'image-p-hbzllm-1h','image-p-tybhc-500ml',
              'image-p-nfdfsymlhc-500ml',
              'image-p-wqmrcptz-300ml',
              'image-p-Hbytfspg-5g',
              'image-big-p-scppg-2l'
             ]

class_id_blacklist = [10,11,12]

# #######################
# Util
# #######################
import urllib
import numpy as np


def load_image_from_url(img_url):
    savepath = './tmp/img.jpg'

    # load from url
    data = urllib.urlopen(img_url).read()

    # cv2 format transfer
    img = np.asarray(bytearray(data),dtype="uint8")
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)

    # save image to ./tmp
    abspath = os.path.abspath(savepath)
    if os.path.exists(abspath):
        os.system('rm %s'%abspath)

    cv2.imwrite(abspath,img)

    return img,abspath

def sec_to_hms(sec):
    hour = int(sec / 3600)
    minute = int((sec - 3600 * hour) / 60)
    second = int((sec - 3600 * hour - 60 * minute))

    return hour,minute,second

def load_image_from_url_no_cv2(img_url,flag=True):
    
    save_tmp_path = './tmp/img.jpg'
    abspath1 = os.path.abspath(save_tmp_path)
        # load from url
    urllib.urlretrieve(img_url,abspath1)
    
    if(flag):
        save_img_path = './images/'+time.strftime("%y_%m_%d", time.localtime())+'/'
        if not os.path.exists(save_img_path):
          os.mkdir(save_img_path)
        save_img =save_img_path+time.strftime("%H_%M_%S", time.localtime())+'_'+str(datetime.datetime.now().microsecond)+'.jpg'
        abspath2 = os.path.abspath(save_img)
        urllib.urlretrieve(img_url,abspath2)

    return abspath1

# #######################
# Goods Detect
# #######################
model_cfg_path = os.path.join(wd, 'material', 'cfg', 'missfresh-cls26-yolo-voc-800.cfg')
model_weights_path = os.path.join(wd, 'material', 'yolo_models', 'missfresh-mix-yolo-voc-800', 'yolo-voc-800_24000.weights')
meta_path = os.path.join(wd, 'material', 'cfg', '%s' % data_info)
# --init detector
net = load_net(model_cfg_path, model_weights_path, 0)
meta = load_meta(meta_path)

mFeature_Extractor=Feature_Extractor()
# Get Feature Sets
mVideo_Feature_Sets = Video_Feature_Sets()
mVideo_Feature_Sets.load(default_video_feature_sets_path)

def goods_detect_urls(
        img_urls,
        yolo_cfg_path=model_cfg_path,
        yolo_weights_path=model_weights_path,
        good_info_path=meta_path,
        conf_thres=[0.7]
):
    """
    Performing goods detection on online images.

    :param img_urls:
    :param yolo_cfg_path:
    :param yolo_weights_path:
    :param good_info_path:
    :param conf_thres:
    :return:
    """
    goods_det_results_dict = {}

    for url in img_urls:

        det_result = []

        #_,im_path = load_image_from_url(url)
        im_path = load_image_from_url_no_cv2(url)

        im_path = os.path.abspath(im_path)
        image= cv2.imread(im_path)
        arr = image.shape
        width = arr[1]
        height = arr[0]  

        res = detect(net, meta, im_path, thresh=0.2)

        # parse result
        for line in res:
            cls_name = line[0]
            cls = classes.index(cls_name)
            prob = line[1]
            

            if prob<0.65:
              continue
            bb = line[2]
            # convert bb

            x = bb[0] / width
            y = bb[1] / height
            w = bb[2] / width
            h = bb[3] / height
            
            
            box1 = (max([0,width*x-width*w/2]), max([0,height*y-height*h/2]), min([width,width*w]), min([height,height*h]))
        
            array_box1 = np.array(box1)
            array_box1 = array_box1.astype(int)
            roii = image[array_box1[1]:array_box1[1]+array_box1[3],array_box1[0]:array_box1[0]+array_box1[2]]
            good_id,good_scaore=content_base_video_retrieval(roii,mFeature_Extractor,mVideo_Feature_Sets)
            if good_id==None:
              continue
            cls = int(good_id[3:5])-1
            cls_id = classes_id[cls] 
            if cls not in class_id_blacklist:
                if(len(conf_thres)==1 and prob>=conf_thres[0]):
                    det_result.append([cls_id,prob])
                elif(len(conf_thres)==len(classes)):
                    det_result.append([cls_id,prob])
                elif(prob<conf_thres[cls] or prob<conf_thres[0]):
                    continue
                else:
                    print 'error'

        goods_det_results_dict[url] = det_result
	
    #print goods_det_results_dict

    return goods_det_results_dict

def goods_detect_urls_yi_plus_local_test(
        img_urls,
        yolo_cfg_path=model_cfg_path,
        yolo_weights_path=model_weights_path,
        good_info_path=meta_path,
        conf_thres=[0.7]
):
    """
    Performing goods detection on online images.

    :param img_urls:
    :param yolo_cfg_path:
    :param yolo_weights_path:
    :param good_info_path:
    :param conf_thres:
    :return:
    """
    for url in img_urls:

        #im_path = load_image_from_url_no_cv2(url,False)

        #im_path = os.path.abspath(im_path)
        im_path=url
        image= cv2.imread(im_path)
        arr = image.shape
        width = arr[1]
        height = arr[0]  
        
        res = detect(net, meta, im_path, thresh=0.2)

        cls_arr=[x+1 for x in range(26)]
        num_cls=[0 for x in range(26)]

        # parse result
        for line in res:
            
            prob = line[1]
            if prob<0.65:
              continue
            bb = line[2]
            # convert bb

            x = bb[0] / width
            y = bb[1] / height
            w = bb[2] / width
            h = bb[3] / height
            box1 = (max([0,width*x-width*w/2]), max([0,height*y-height*h/2]), min([width,width*w]), min([height,height*h]))
            print box1
            array_box1 = np.array(box1)
            array_box1 = array_box1.astype(int)
            #roii = image[max([0,array_box1[1]]):min([width,array_box1[1]+array_box1[3]]),max([0,array_box1[0]]):min([height,array_box1[0]+array_box1[2]])] 
            roii = image[array_box1[1]:array_box1[1]+array_box1[3],array_box1[0]:array_box1[0]+array_box1[2]]  
            cv2.imwrite('./tmp/img_1.jpg',roii)
            stime = time.time()       
            good_id,good_scaore=content_base_video_retrieval(roii,mFeature_Extractor,mVideo_Feature_Sets)
            etime = time.time()
            dur_sec = etime - stime
            print 'Total io time: %f'%dur_sec
            if good_id==None:
              continue            

            #good_id=content_base_video_retrieval('./tmp/img_1.jpg')

            
            cls = int(good_id[3:5])-1
            if cls not in class_id_blacklist:
                if(len(conf_thres)==1 and prob>=conf_thres[0]):
                    num_cls[cls]+=1
                    print cls_name
                    print prob
                elif(len(conf_thres)==len(classes)): #and prob>=conf_thres[cls]):
                    num_cls[cls]+=1
                    print good_id[3:5]
                    print prob
                elif(prob<conf_thres[cls] or prob<conf_thres[0]):
                    continue
                else:
                    print 'error'

        merg=[cls_arr,num_cls]
        merg=map(list,zip(*merg))
        print merg



if __name__ == "__main__":

    #urls = ['http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/ba575e54-3e22-11e8-a283-d0817abd9fdc.jpg','http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/bbc399cc-3e22-11e8-b2da-d0817abd9fdc.jpg','http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/c20a8b62-3e22-11e8-845b-d0817abd9fdc.jpg','http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/c2c53f52-3e22-11e8-9d13-d0817abd9fdc.jpg']
    #urls =['http://yijiaadplatform.dressplus.cn/AdPlatform/images/2018-04-12/ba575e54-3e22-11e8-a283-d0817abd9fdc.jpg']
    urls = ['/home/tanfulun/workspaces/Project/eval_yolo_detection/tmp/img1.jpg']
    #each_thres=[.2,.3,.7,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2,.2]
    each_thres=[.65,.65,.65,.65,.65]
    goods_detect_urls_yi_plus_local_test(urls,conf_thres=each_thres)


