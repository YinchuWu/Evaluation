import os,sys
import numpy as np
from numpy import linalg as LA
import cv2

os.environ['GLOG_minloglevel'] = '2'
import caffe

from conf import *

# --Caffe Model
caffe_model_path = os.path.join(wd,'material','models','yi+shopping.caffemodel')
caffe_prototxt_path = os.path.join(wd,'material','models','yi+shopping.prototxt')

if not os.path.exists(caffe_model_path):
    print('Caffe model file missing!')
    exit()

if not os.path.exists(caffe_prototxt_path):
    print('Caffe prototxt file missing!')
    exit()

# --ILSVRC2012 Mean file
ilsvrc2012_mean_npy_path = os.path.join(wd,'material','models','ilsvrc_2012_mean.npy')
if not os.path.exists(ilsvrc2012_mean_npy_path):
    print('ilsvrc12 mean file missing!')
    exit()


class Feature_Extractor:
    """
    Extract Features from a input image.

    Using caffe model

    """
    def __init__(self):

        # load ilsvrc12 mean
        ilsvrc12_mean = np.load(ilsvrc2012_mean_npy_path)
        ilsvrc12_mean = ilsvrc12_mean.mean(1).mean(1)
        self.ilsvrc12_mean = ilsvrc12_mean

        # init caffe model
        net = caffe.Net(caffe_prototxt_path,  # defines the structure of the model
                        caffe_model_path,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)
        self.net = net

        # init image transformer
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', self.ilsvrc12_mean)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        self.transformer = transformer

    def extract(self,image_path):#image_path
        """
        Extract Features from a input image

        :param image_path: path of input image
        :return:
        """

        img = caffe.io.load_image(image_path)
        
        #image1=cv2.imread(caffe_root + 'examples/images/cat.jpg')  
        #img=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)  
        #img=img/255. 
        

        transformed_image = self.transformer.preprocess('data', img)
        self.net.blobs['data'].data[...] = transformed_image
        ft = self.net.forward()
        ft = np.squeeze(ft['pool5/7x7_s1'])
        ft = ft / LA.norm(ft)
        return ft

