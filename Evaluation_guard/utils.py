import numpy as np
import os
import cv2
import glob

import pdb

# ################################
# DataSet
# ################################
train_info = 'models_4w_lr_0.001'
test_info = 'yolo-voc'
prefix_default = 'yolo-voc_'
#iterations_default = [2000,5000,10000,20000,25000,30000,34000,36000,38000,40000]
#iterations_default = [2000,5000,10000,20000,25000,30000,34000,36000,38000]
iterations_default = [25000]
sets_default = ['train','val']

def make_dataset(prefix=prefix_default,sets=sets_default,iterations=iterations_default,train_info=train_info,test_info=test_info):
    '''
    Make Datasets with full info

    :param prefix:
    :param sets:
    :param iterations:
    :param train_info:
    :param test_info:
    :return:
    '''
    DataSets = []

    for set in sets:
        for itr in iterations:
            DataSets.append(('%s%d'%(prefix,itr),'%s'%set,'%s'%train_info,'%s'%test_info))

    return DataSets

def make_dataset_model_only(prefix=prefix_default,iterations=iterations_default,test_info=test_info):
    '''
    Make Datasets with "model info" and "predict result info"

    :param prefix: model weights prefix
    :param iterations:
    :param test_info: predict results info
    :return:
    '''
    DataSets = []

    for itr in iterations:
        DataSets.append(('%s%d'%(prefix,itr),'%s'%test_info))

    return DataSets

# ################################
# Format Change
# ################################
def convert_bb_format(boxes):
    '''from (cx,cy,w,h)-->(x1,y1,x2,y2)

    :param boxes:
    :return:
    '''
    new_boxes = []

    for box in boxes:
        b_w = box[2]
        b_h = box[3]
        c_x = box[0]
        c_y = box[1]

        x1 = (max([0., (c_x - 0.5 * b_w)]))
        x2 = (min([1., (c_x + 0.5 * b_w)]))
        y1 = (max([0., (c_y - 0.5 * b_h)]))
        y2 = (min([1., (c_y + 0.5 * b_h)]))

        new_boxes.append(np.array([x1,y1,x2,y2]))

    new_boxes = np.array(new_boxes)

    return new_boxes

# ################################
# Data tranfer
# ################################
def load_weights_from_server(dataset):
    for ds in dataset:
        os.system('scp -P 9502 tanfulun@gpu.dress.plus:/mnt/nas/tanfulun/Project/darknet/tfl/goods-id/backup/yolo-voc_38000.weights ./')


# ################################
# plot
# ################################
def plot_bb_on_img(im,bb,color,bbThickness=2,textSize=4,textThickness=4,info=''):
    h, w, c = im.shape

    for idx, box in enumerate(bb):

        b_w = box[2] * w
        b_h = box[3] * h
        c_x = box[0] * w
        c_y = box[1] * h

        x1 = int(max([0, (c_x - 0.5 * b_w)]))
        x2 = int(min([w, (c_x + 0.5 * b_w)]))
        y1 = int(max([0, (c_y - 0.5 * b_h)]))
        y2 = int(min([h, (c_y + 0.5 * b_h)]))

        cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=bbThickness)
        cv2.putText(im, '%s' % info, (x1, y1+100), cv2.FONT_HERSHEY_SIMPLEX, textSize, color, thickness=textThickness)

    return im

# ################################
# Utils
# ################################
def get_file_name_in_dir(in_dir,out_path,suffix='*.jpg'):
    '''
    Get the file names in "in_dir"

    :param in_dir: file dir
    :param out_path: save path
    :param suffix: file suffix
    :return: None
    '''
    filelist = glob.glob(os.path.join(in_dir, suffix))
    #pdb.set_trace()
    f = open(out_path,'w')
    for line in filelist:
        name = line.split('/')[-1].split('.')[0]
        f.write(name+'\n')
    f.close()

    return 0

def get_file_full_path_in_dir(files_dir,out_path,suffix='*.jpg'):
    '''
    Get file full path from directory "files_dir"

    :param files_dir:
    :param out_path:
    :param suffix:
    :return:
    '''
    filelist = glob.glob(os.path.join(files_dir, suffix))
    f = open(out_path, 'w')
    for line in filelist:
        f.write(line + '\n')
    f.close()

def change_img_format(in_dir,out_dir,in_suffix='*.bmp'):
    #pdb.set_trace()
    filelist = glob.glob(os.path.join(in_dir, in_suffix))

    for imgpath in filelist:
        im = cv2.imread(imgpath)
        imgname = imgpath.split('/')[-1].split('.')[0]

        # remove "blankspace" in image names
        imgname = imgname.split(' ')
        imgname = ''.join(imgname)

        outpath = os.path.join(outdir,imgname+'.jpg')
        cv2.imwrite(outpath,im)
        #pdb.set_trace()

    pass

def balance_lightness(name,clahe):
    in_img_dir='/mnt/disk1/lvsikai/missfresh/Data/7MissFreshSmartShelf_Exp/JPEGImages/'
    out_img_dir='/mnt/disk1/lvsikai/missfresh/Data/8MissFreshSmartShelf_Exp/JPEGImages/'

    in_image_path=os.path.join(in_img_dir,name)
    out_image_path=os.path.join(out_img_dir,name)

    bgr = cv2.imread(in_image_path)
    
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    
    lab_planes = cv2.split(lab)
    
    #clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    lab = cv2.merge(lab_planes)
    
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    cv2.imwrite(out_image_path,bgr)

# ################################
# Test
# ################################
if __name__ == "__main__":

    # get file names/full path in dir
    '''
    imgdir = '/home/tfl/workspace/dataSet/MissFreshSmartShelf/JPEGImages'
    outdir = '/home/tfl/workspace/dataSet/MissFreshSmartShelf'
    get_file_name_in_dir(imgdir,outdir+'/test_name.txt')
    get_file_full_path_in_dir(imgdir,outdir+'/test_full.txt')
    exit()

    # change images format
    imgdir = '/home/tfl/workspace/dataSet/MissFreshSmartShelf/SelectedTestsetImages'
    outdir = '/home/tfl/workspace/dataSet/MissFreshSmartShelf/JPEGImages'
    change_img_format(imgdir,outdir)
    '''
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    filelist = glob.glob('/mnt/disk1/lvsikai/missfresh/Data/7MissFreshSmartShelf_Exp/JPEGImages/*.jpg')
    for line in filelist:
        name = line.split('/')[-1]
        if(name[-5]=='p' or name[0]=='g'):
            continue
        balance_lightness(name,clahe)