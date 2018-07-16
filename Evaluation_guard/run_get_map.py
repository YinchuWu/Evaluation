# coding: utf-8

import os
import sys
from utils import *
import cv2
import numpy as np
import pdb

import evaluation as my_eval


from conf import *

"""
Get object detection MAPs
"""

# #############
# Functions
# #############


def run_get_map(dataset, model_name, test_info):
    '''
    run testing on a "dataset"
    :param dataset: train/val
    :param model_name: yolo-model names
    :param test_info: folder of predicting results
    :return:
    '''

    # --get list of image names
    # path of train.txt/val.txt
    dataset_file = os.path.join(
        dataset_dir, 'ImageSets', 'Main', '%s.txt' % dataset)
    f = open(dataset_file)
    img_names = f.readlines()
    f.close()

    img_names = [x.strip() for x in img_names]

    # --get predict results dir
    predict_results_first_lever_dir = os.path.join(
        wd, 'results', 'predict', test_info)
    if not os.path.exists(predict_results_first_lever_dir):
        print("Folder not exist!")
        print(predict_results_first_lever_dir)
        exit()

    predict_results_second_lever_dir = os.path.join(
        predict_results_first_lever_dir, model_name)
    if not os.path.exists(predict_results_second_lever_dir):
        os.mkdir(predict_results_second_lever_dir)

    predict_results_dir = os.path.join(
        predict_results_second_lever_dir, dataset)
    if not os.path.exists(predict_results_dir):
        print("Folder not exist!")
        print(predict_results_dir)
        exit()

    # --get GT dir
    gt_dir = os.path.join(dataset_dir, 'dk_labels')

    # --calculate AP
    mAP = 0  # all cls map

    fg_bg_AP = 0  # fg-bg map

    AP = np.zeros(len(classes))
    cls_img_count = np.zeros(len(classes))  # img numbers of each class
    img_with_more_than_one_cls = []

    for name in img_names:
        if(name[0] == 'g' or len(name) > 27):
            continue
        print(name)
        # pdb.set_trace()

        # list of predict results
        predict_file = open(os.path.join(predict_results_dir, '%s.txt' % name))
        pt_list = predict_file.readlines()
        predict_file.close()
        pt_list = [x.strip().split(' ') for x in pt_list]

        pred_boxes = [[float(x[2]), float(x[3]), float(
            x[4]), float(x[5])] for x in pt_list]
        pred_class_ids = [int(x[0]) for x in pt_list]
        pred_scores = [float(x[1]) for x in pt_list]

        # list of GT
        gt_file = open(os.path.join(gt_dir, '%s.txt' % name))
        gt_list = gt_file.readlines()
        gt_file.close()
        gt_list = [x.strip().split(' ') for x in gt_list]

        gt_boxes = [[float(x[1]), float(x[2]), float(x[3]),
                     float(x[4])] for x in gt_list]
        gt_class_ids = [int(x[0]) for x in gt_list]

        # to np.array()
        # pdb.set_trace()
        gt_boxes = np.array(gt_boxes)
        gt_class_ids = np.array(gt_class_ids)
        pred_boxes = np.array(pred_boxes)
        pred_class_ids = np.array(pred_class_ids)
        pred_scores = np.array(pred_scores)
        # pdb.set_trace()

        # convert bb format to (x1,y1,x2,y2)
        gt_boxes = convert_bb_format(gt_boxes)

        pred_boxes = convert_bb_format(pred_boxes)
        if(len(gt_boxes.shape) != 2 or len(pred_boxes.shape) != 2):
            continue

        #
        img_mAP, _, _, _ = my_eval.compute_ap(
            gt_boxes,
            gt_class_ids,
            pred_boxes,
            pred_class_ids,
            pred_scores
        )
        fg_bg_AP = fg_bg_AP + img_mAP
        #
        # pdb.set_trace()
        cls_num = 0
        for cls in range(len(classes)):
            if cls in gt_class_ids:
                cls_num = cls_num + 1
                # pdb.set_trace()
                idx = np.where(gt_class_ids == cls)[0]
                pt_idx = np.where(pred_class_ids == cls)[0]

                gt_b = gt_boxes[idx]
                gt_c_ids = gt_class_ids[idx]
                pt_b = pred_boxes[pt_idx]
                pt_c_ids = pred_class_ids[pt_idx]
                pt_s = pred_scores[pt_idx]

                cls_AP, _, _, _ = my_eval.compute_ap(
                    gt_b,
                    gt_c_ids,
                    pt_b,
                    pt_c_ids,
                    pt_s
                    # pred_boxes,
                    # pred_class_ids,
                    # pred_scores
                )
                print (cls_AP)
                # pdb.set_trace()
                AP[cls] = AP[cls] + cls_AP
                cls_img_count[cls] = cls_img_count[cls] + 1

        if cls_num > 1:
            img_with_more_than_one_cls.append(name)

    for i in range(len(AP)):
        if(0 == AP[i]):
            AP[i] = 1.
            cls_img_count[i] = 1.
    AP = AP / cls_img_count
    mAP = sum(AP) / len(AP)
    fg_bg_AP = fg_bg_AP / len(img_names)

    result_f = open(os.path.join(
        predict_results_second_lever_dir, '%s_mAP.txt' % dataset), 'w')
    result_f.write('mAP:%f\n' % mAP)
    result_f.write('fg_bg_AP:%f\n' % fg_bg_AP)
    for idx, item in enumerate(classes):
        result_f.write('%s:%f\n' % (item, AP[idx]))

    result_f.close()

    # pdb.set_trace()


if __name__ == "__main__":
    # {(model_name,dataset_name),...,...}

    # prefix of yolo-model file (.weight)
    ds_prefix = 'yolo-voc-800_'

    # folder of predicting results (default dir: ./results/predict/)
    ds_test = 'missfresh-yolo-voc-800-0523'

    # DataSets Type(train/val)
    sets = ['val']  # , 'train'
    #sets = ['test']

    # checkpoints of models
    # ,24000,26000,28000,30000],10000,16000,18000,20000,22000,24000
    checkpoints = [8000, 10000, 12000, 14000, 16000, 18000, 20000]

    DataSets = make_dataset(
        prefix=ds_prefix, test_info=ds_test, sets=sets, iterations=checkpoints)

    # get MAPs
    for ds in DataSets:
        model_name = ds[0]
        dataset_name = ds[1]
        test_info = ds[3]

        run_get_map(dataset_name, model_name, test_info)
