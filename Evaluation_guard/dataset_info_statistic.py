import numpy as np
import os
import matplotlib.pyplot as plt

#from conf import *

# #############
# Functions
# #############

MAX_BB_NUMBER = 60
dataset_dir = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\7MissFreshSmartShelf_Exp'
classes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26']

def run_analyse():
    #
    bb_bin = {}
    # get label file
    label_dir = os.path.join(dataset_dir,'dk_labels')
    if not os.path.exists(label_dir):
        print ('label file missing!')
        exit()

    # get dataset
    imageset_dir = os.path.join(dataset_dir,'ImageSets','Main')
    if not os.path.exists(imageset_dir):
        print ('dataset dir missing!')
        exit()

    datasets = ['train','val']
    for ds in datasets:
        ds_file = os.path.join(imageset_dir,'%s.txt'%ds)
        if not os.path.exists(ds_file):
            print ('%s.txt file missing!'%ds)
            exit()
        #
        f = open(ds_file)
        img_names = f.readlines()
        f.close()
        #
        img_names = [x.strip() for x in img_names]
        bb_counts = np.zeros(len(classes))
        multi_label_cnt = 0

        max_bb_per_img = 0  # max number of bb in one image
        bb_per_img_bin = np.zeros(MAX_BB_NUMBER)

        for name in img_names:
            # print name
            #muli_label = False
            #
            #if(name[0]=='M'):
            if name[-1]=='p':
                continue
            gt_file = open(os.path.join(label_dir, '%s.txt' % name))
            gt_list = gt_file.readlines()
            gt_file.close()
            gt_list = [x.strip().split(' ') for x in gt_list]

            gt_class_ids = [int(x[0]) for x in gt_list]
            #
            for idx in gt_class_ids:
                bb_counts[idx] = bb_counts[idx] + 1

            if len(set(gt_class_ids))>1:
                multi_label_cnt = multi_label_cnt + 1

            #
            bb_sum = len(gt_class_ids)
            bb_per_img_bin[bb_sum] = bb_per_img_bin[bb_sum] + 1
            if bb_sum>max_bb_per_img:
                max_bb_per_img = bb_sum

        print ("\n# %s"%ds)
        print ("max_bb_per_img:%d"%max_bb_per_img)

        for idx,item in enumerate(bb_counts):
            print ('--%s: %d'%(classes[idx],item))

        bb_bin[ds] = bb_per_img_bin

    # plot bar
    _,(ax1,ax2) = plt.subplots(1, 2, sharex=True,sharey=True)
    x = range(MAX_BB_NUMBER)

    # ax2
    key = datasets[0]
    val = bb_bin[key]

    ax1.bar(x,val)
    ax1.set_label(key)
    # ax2
    key = datasets[1]
    val = bb_bin[key]

    ax2.bar(x, val)
    ax2.set_label(key)

    plt.show()

    pass

if __name__=="__main__":
    run_analyse()