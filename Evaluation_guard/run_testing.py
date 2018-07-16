#coding: utf-8

import os,sys
from darknet import *
import cv2
import pdb
from utils import *
import multiprocessing as mp

from conf import *

from image_feature_extractor import Feature_Extractor
from Make_Video_Feature_Sets import Video_Feature_Sets
#from Search_Video_by_Image import content_base_video_retrieval

import numpy as np


"""
Run object detection on images, and get results
"""

# #############
# Functions
# #############

def run_test(dataset,model_name,train_info,test_info,thres=0.001):
    '''
    run testing on a "dataset"

    :param dataset:
    :param model_name:
    :param train_info:
    :param test_info:
    :param thres: confidence for a bb
    :return:
    '''
    #pdb.set_trace()
    
    model_cfg_path = os.path.join(wd,'material','cfg','%s.cfg'%test_info)
    model_weights_path = os.path.join(wd,'material','yolo_models','%s'%train_info,'%s.weights'%model_name)
    meta_path = os.path.join(wd,'material','cfg','%s'%data_info)#'/mnt/disk1/lvsikai/missfresh/eval_yolo_detection/material/cfg/missfresh.names'#

    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt'%dataset)
    
    # predict results
    predict_results_dir = os.path.join(wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir, model_name)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir,dataset)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    # --get testing data
    fp = open(dataset_file,'r')
    img_names = fp.readlines()
    fp.close()

    img_names_list = [x.strip() for x in img_names]
    #print img_names_list

    img_paths = []
    #img_labels_path = []
    
    for name in img_names_list:
        #if(name[-1]=='p'):
          #continue
        path = os.path.join(dataset_dir, 'JPEGImages','%s.jpg'%name)
        #label_path = os.path.join(dataset_dir, 'dk_labels','%s.txt'%name)
        if not os.path.isfile(path):
            print ("file missing!")
            print (path)
            exit()
#        if not os.path.isfile(label_path):
#            print ("file missing!")
#            print (label_path)
#            exit()
        img_paths.append(path)
        #img_labels_path.append(label_path)

    # --run testing
    #pdb.set_trace()

    # load yolo-model
    net = load_net(model_cfg_path, model_weights_path, 0)
    # load Experiment information
    meta = load_meta(meta_path)

    for idx,item in enumerate(img_names_list):
        if(item[-1]=='p'):
          continue
	print len(img_paths)
	print idx
        print ('Processing '+item)

        result_txt_path = os.path.join(predict_results_dir,'%s.txt'%item)

        result_f = open(result_txt_path,'w')

        res = detect(net, meta, img_paths[idx],thresh=thres)

        im = cv2.imread(img_paths[idx])

        (im_h,im_w,im_c) = im.shape

        for line in res:
            cls_name = line[0]
            cls = classes.index(cls_name)
            prob = line[1]
            bb = line[2]
            # convert bb

            x = bb[0] / im_w
            y = bb[1] / im_h
            w = bb[2] / im_w
            h = bb[3] / im_h

            result_f.write('%d %f %f %f %f %f\n'%(cls,prob,x,y,w,h))

        result_f.close()

def run_test_on_testset(dataset,model_name,train_info,test_info,thres=0.001):
    '''
    run testing on a "dataset"
    :param dataset:
    :param model_name:
    :return:
    '''

    model_cfg_path = os.path.join(wd, 'material', 'cfg', '%s.cfg' % test_info)
    model_weights_path = os.path.join(wd, 'material', 'yolo_models', '%s' % train_info, '%s.weights' % model_name)#train_info:model; model_name:test
    meta_path = os.path.join(wd, 'material', 'cfg', '%s' % data_info)

    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '%s.txt'%dataset)
    

    # predict results
    predict_results_dir = os.path.join(wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir, model_name)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir,dataset)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    # --get testing data
    fp = open(dataset_file,'r')
    img_names = fp.readlines()
    fp.close()

    img_names_list = [x.strip() for x in img_names]

    img_paths = []
    img_labels_path = []
    for name in img_names_list:
        path = os.path.join(dataset_dir, 'JPEGImages','%s.jpg'%name)
        if not os.path.isfile(path):
            print ("file missing!")
            print (path)
            exit()

        img_paths.append(path)

    # --run testing
    #pdb.set_trace()
    net = load_net(model_cfg_path, model_weights_path, 0)
    meta = load_meta(meta_path)

    for idx,item in enumerate(img_names_list):

        print ('Processing '+item)

        result_txt_path = os.path.join(predict_results_dir,'%s.txt'%item)

        result_f = open(result_txt_path,'w')

        res = detect(net, meta, img_paths[idx],thresh=thres)

        im = cv2.imread(img_paths[idx])

        (im_h,im_w,im_c) = im.shape

        for line in res:
            cls_name = line[0]
            cls = classes.index(cls_name)
            prob = line[1]
            bb = line[2]
            # convert bb

            x = bb[0] / im_w
            y = bb[1] / im_h
            w = bb[2] / im_w
            h = bb[3] / im_h

            result_f.write('%d %f %f %f %f %f\n'%(cls,prob,x,y,w,h))

        result_f.close()

def run_test_on_testset_one(dataset,model_name,train_info,test_info,image_path,thres=0.001):
    '''
    run testing on one image
    :param dataset:
    :param model_name:
    :return:
    '''
    
    model_cfg_path = os.path.join(wd, 'material', 'cfg', '%s.cfg' % test_info)
    model_weights_path = os.path.join(wd, 'material', 'yolo_models', '%s' % train_info, '%s.weights' % model_name)#train_info:model; model_name:test
    meta_path = os.path.join(wd, 'material', 'cfg', '%s' % data_info)

    # predict results
    predict_results_dir = os.path.join(wd,'results', 'predict', test_info)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir, model_name)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    predict_results_dir = os.path.join(predict_results_dir,dataset)
    if not os.path.exists(predict_results_dir):
        os.mkdir(predict_results_dir)

    # --run testing
    #pdb.set_trace()
    im = cv2.imread(image_path)
    net = load_net(model_cfg_path, model_weights_path, 0)
    meta = load_meta(meta_path)
    
    mFeature_Extractor=Feature_Extractor()
    # Get Feature Sets
    mVideo_Feature_Sets = Video_Feature_Sets()
    mVideo_Feature_Sets.load(default_video_feature_sets_path)    

    #for idx,item in enumerate(img_names_list):
    
    print ('Processing '+image_path)

    result_txt_path = os.path.join(predict_results_dir,'a.txt')

    result_f = open(result_txt_path,'w')
    res = detect(net, meta, image_path,thresh=thres)

    (im_h,im_w,im_c) = im.shape

    for line in res:
        cls_name = line[0]
        cls = classes.index(cls_name)
        prob = line[1]
        bb = line[2]
        # convert bb
        x = bb[0] / im_w
        y = bb[1] / im_h
        w = bb[2] / im_w
        h = bb[3] / im_h
        
        cls=class_trasfer(im,x,y,w,h,im_h,im_w,mFeature_Extractor,mVideo_Feature_Sets)

        result_f.write('%d %f %f %f %f %f\n'%(cls,prob,x,y,w,h))

    result_f.close()
    
def class_trasfer(img,x,y,w,h,height,width,mFeature_Extractor,mVideo_Feature_Sets):
    
    box1 = (max([0,width*x-width*w/2]), max([0,height*y-height*h/2]), min([width,width*w]), min([height,height*h]))
    array_box1 = np.array(box1)
    array_box1 = array_box1.astype(int)
    roii = img[array_box1[1]:array_box1[1]+array_box1[3],array_box1[0]:array_box1[0]+array_box1[2]]
    good_id,good_scaore=content_base_video_retrieval(roii,mFeature_Extractor,mVideo_Feature_Sets)
    return good_id

def content_base_video_retrieval(query,mFeature_Extractor,mVideo_Feature_Sets):
    """
    Main function of CBVR

    :param query: Query,should be a 'path' or 'dir'
    :return:
    """

    # Init Feature Extractor
    mFeature_Extractor = mFeature_Extractor
  
    result_frame_info,result_frame_score = _video_search_by_image(mFeature_Extractor, mVideo_Feature_Sets, query,top=1)
    
    return result_frame_info,result_frame_score

def _video_search_by_image(Extractor,FeatureSets,Query_Image,conf_thres=0.5,top=10):#Query_Image_Path
    """
    Searching by Image file

    :param Extractor:
    :param FeatureSets:
    :param Query_Image_Path:
    :param conf_thres:
    :param top:
    :return:
    """
    # --Get query image feature
    ft = Extractor.extract(Query_Image)

    # --Get DB
    db = FeatureSets.Feat_Sets
    frame_feats = db['frame_feature']
    frame_infos = db['frame_info']

    # --list->np.array()
    ft = np.array(ft)
    frame_feats = np.array(frame_feats)

    # --Matching
    scores = np.dot(ft, frame_feats.T)
    scores = np.array(scores)
    rank_ID = np.argsort(-scores)
    rank_score = scores[rank_ID]

    # --Get top ranking results
    result_frame_info = []
    result_score = []

    for k in range(0,top):
      if rank_score[k]>conf_thres:
        result_frame_info.append(frame_infos[rank_ID[k]])
        result_score.append(rank_score[k])
        return frame_infos[rank_ID[k]],rank_score[k]
    return None,None

if __name__ == "__main__":
    # {(model_name,dataset_name),...,...}

    # prefix of yolo-model file (.weight)
    ds_prefix = 'yolo-voc-800_'
    # folder of yolo model files (default dir: ./yolo_models/)
    ds_train = 'missfresh-yolo-voc-800-0503'
    # folder of predicting results (default dir: ./results/predict/)
    ds_test = 'missfresh-yolo-voc-800-0514'


    # DataSets Type(train/val/test)
    #sets = ['val','train','item','test']
    sets = ['item']#,'val']
    #absolute path of image
    item='/mnt/disk1/lvsikai/missfresh/Data/test/img07.jpg'
    # checkpoints of models
    checkpoints = [24000]
    # Creating Dataset information
    DataSets = make_dataset(prefix=ds_prefix,train_info=ds_train,test_info=ds_test,sets=sets,iterations=checkpoints)

    #pdb.set_trace()

    # get predict results
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        train_info = ds[2]
        test_info = ds[3]

        #run_test_on_testset(dataset_name,model_name,train_info,test_info)

        if dataset_name=='test':
            #针对未标记的testSet
            sub_process = mp.Process(target=run_test_on_testset,args=(dataset_name,model_name,train_info,test_info))
        elif dataset_name=='item':
            sub_process = mp.Process(target=run_test_on_testset_one,args=(dataset_name,model_name,train_info,test_info,item))
        else:
            #针对已标记的trainSet/valSet
            sub_process = mp.Process(target=run_test, args=(dataset_name, model_name,train_info,test_info))
        sub_process.start()
        sub_process.join()
        sub_process.terminate()




