# -*- coding: utf-8 -*-
import numpy as np
import os,sys
import glob
import cv2

os.environ['GLOG_minloglevel'] = '2'
import caffe
from image_feature_extractor import Feature_Extractor
from Make_Video_Feature_Sets import Video_Feature_Sets
#from Get_Washed_Video_Frame import get_key_frame_and_info
#from util import sec_to_hms

import pdb

from conf import *

# set gpu mode
caffe.set_mode_gpu()

output = []
res_feats = []


def content_base_video_retrieval(query,mFeature_Extractor,mVideo_Feature_Sets):
    """
    Main function of CBVR

    :param query: Query,should be a 'path' or 'dir'
    :return:
    """

    # Init Feature Extractor
    mFeature_Extractor = mFeature_Extractor


    # get query type.
    # 1)If query is a image,suffix should be the suffix of it;
    # 2)If query is a dir,suffix=''
    #suffix = os.path.splitext(query)[1]
    
    #
    #if suffix == ('.jpg' or '.png'):    
    result_frame_info,result_frame_score = _video_search_by_image(mFeature_Extractor, mVideo_Feature_Sets, query,top=1)
    
    return result_frame_info,result_frame_score
        
    
    '''
    print('\n')
    print('-----Show Top%d Results-----' % topN)
    for idx,data in enumerate(res_list):
        print('#%d Name:%s, %d:%d:%d'%(idx+1,data[0],data[1][0],data[1][1],data[1][2]))
    print('\n')
    '''
    '''
    # plot result
    should_plot_result = False
    show_w = 400
    show_h = 300
    final_img = []
    if should_plot_result:
        img_query = cv2.imread(query)
        img_query = cv2.resize(img_query,(show_w,show_h),interpolation=cv2.INTER_CUBIC)

        final_img = img_query

        db_image_dir = os.path.join(wd,'demo','video_key_frames','Battle-20170930','images')
        for frame_info in result_frame_info:
            pdb.set_trace()
            f_name = 'frame_'+frame_info.split('_')[-1]+'.jpg'
            img = cv2.imread(os.path.join(db_image_dir,f_name))
            img = cv2.resize(img, (show_w, show_h), interpolation=cv2.INTER_CUBIC)
            final_img = np.hstack((final_img,img))

        cv2.imshow('Top%d Results'%topN,final_img)

        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
    '''
    '''
    else:
        v_name,v_st_sec,v_end_sec = _video_search_by_video_clip(mFeature_Extractor, mVideo_Feature_Sets, query)
        v_st_h,v_st_m,v_st_s = sec_to_hms(v_st_sec)
        v_end_h, v_end_m, v_end_s = sec_to_hms(v_end_sec)
    
        print('\n')
        print('-----Results-----')
        print('Video Name:%s, from %d:%d:%d to %d:%d:%d' % (v_name, v_st_h,v_st_m,v_st_s, v_end_h, v_end_m, v_end_s))
        print('\n')
    '''

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
    

def _video_search_by_video_clip(Extractor,FeatureSets,Query_Image_dir,conf_thres=0.5,top=10):
    """
    Searching by video-clip

    :param Extractor:
    :param FeatureSets:
    :param Query_Image_dir:
    :param conf_thres:
    :param top:
    :return:
    """

    if not os.path.isdir(Query_Image_dir):
        print('Invalid Query Images Dir ...')
        exit()

    # --Get db video info
    db_video_info = _get_db_video_info(FeatureSets)

    # --Get Washed Video Frames and info of Query Video clip
    query_frames_dir = os.path.join(Query_Image_dir,'images')
    query_info_path = os.path.join(Query_Image_dir,'video_info.txt')

    # get query frames abspath
    key_frame_list = glob.glob(os.path.join(query_frames_dir,'*.jpg'))
    key_frame_list.sort(key=lambda name: int(name.split('_')[-1].split('.')[0])) #sort by frame NO.

    # get query info
    fid = open(query_info_path, 'r')
    query_info = fid.readlines()
    fid.close()

    query_info = [x.strip() for x in query_info]

    query_info_dict = {}
    for line in query_info:
        k = line.split(':')[0]
        v = line.split(':')[1]
        query_info_dict[k] = v

    query_fps = int(query_info_dict['fps'])

    # --Searching
    query_frame_sec_list = []
    res_dict = {}

    for frame_path in key_frame_list:
        # query frame index
        query_frame_idx = frame_path.split('/')[-1].split('.')[0].split('_')[-1]
        query_frame_idx = int(query_frame_idx)

        # query frame time(unit sec)
        sec = (query_frame_idx * 1.0) / (query_fps * 1.0)
        query_frame_sec = int(np.floor(sec))

        query_frame_sec_list.append(query_frame_sec)

        # Matching query image with DB
        res,_ = _video_search_by_image(Extractor, FeatureSets, frame_path,top=50)

        #pdb.set_trace()

        # static one frame result
        for data in res:
            # Matched frame index
            tmp_data = data.split('_')
            db_video_name = tmp_data[0]
            db_video_idx = int(tmp_data[-1])
            # Matched frame time(unit sec)
            db_video_fps = int(db_video_info[db_video_name]['fps'])
            sec = (db_video_idx*1.0)/(db_video_fps*1.0)
            db_video_sec = int(np.floor(sec))

            # time diff between 'Query' and 'Matching frames'
            diff_sec = db_video_sec-query_frame_sec

            if res_dict.has_key(db_video_name):
                res_dict[db_video_name].append(diff_sec)
            else:
                res_dict[db_video_name]=[diff_sec]

    # --Get Query video clip duration(unit sec)
    query_dur_sec = query_frame_sec_list[-1]-query_frame_sec_list[0]

    # --Find max matching time-diff for each video in DB
    res_bin_dict = {}
    for key,val in res_dict.iteritems():
        val_unique = list(set(val))
        max_cnt=0
        max_diff_sec = -1
        for data in val_unique:
            cnt = val.count(data)
            if cnt>max_cnt:
                max_cnt = cnt
                max_diff_sec = data
        res_bin_dict[key] = [max_diff_sec,max_cnt]

    #pdb.set_trace()

    # --Get final results
    match_db_video_name = 'None'
    match_db_sec_diff = -1
    video_st = -1
    video_end = -1

    default_max_cnt = 1 # match frame cnt should be larger than this value
    for key,val in res_bin_dict.iteritems():
        if val[1]>default_max_cnt:
            match_db_video_name = key
            match_db_sec_diff = val[0]

    video_st = max(0,query_frame_sec_list[0]+match_db_sec_diff)
    video_end = video_st+query_dur_sec

    #print('video name:%s, start:%s, end:%s'%(match_db_video_name,video_st,video_end))

    return match_db_video_name,video_st,video_end

def _get_db_video_info(FeatureSets):
    """

    :param FeatureSets:
    :return:
    """
    video_info = {}

    db_video_info = FeatureSets.Feat_Sets['video_info']
    for info in db_video_info:
        v_name = info['Name']
        video_info[v_name] = info

    return video_info



if __name__ == '__main__':
    '''
    # Query Image Info
    query_img_name = 'frame_010899.jpg'
    query_img_dir = '/workspace/Share/data_transfer/video_key_frames/Game_of_Throne/images'
    query_img_path = os.path.join(query_img_dir,query_img_name)

    content_base_video_retrieval(query_img_path)
    '''
    # Query by video
    query_video_dir = wd+'/demo/video_key_frames/Rango-clip-408-435/'

    content_base_video_retrieval(query_video_dir)




