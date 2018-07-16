#coding: utf-8

#-----实验相关配置信息-----#

# working dir(当前代码所在的绝对路径)
wd = 'C:\\Users\\lvsikai\\Desktop\\repositories\\eval_yolo_detection'

# dataSets dir(实验数据存储路径)
dataset_dir = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\4MissFreshSmartShelf_Exp'

# classes(类别名称列表)
classes = ['1','2','3','4','5','6']
#classes = ['1','2','3','4','5']

# 实验数据信息
'''
classes=18 #类别总数
train=/mnt/lvmhdd/tanfulun/workspaces/Data/MissFreshSmartShelf_Exp/train.txt #训练数据路径(废弃)
val=/mnt/lvmhdd/tanfulun/workspaces/Data/MissFreshSmartShelf_Exp/val.txt #测试数据路径(废弃)
names=/home/tfl/workspace/project/YI/goodsid/testing_code/material/cfg/missfresh.names #实验类别名称
backup=/mnt/lvmhdd/tanfulun/workspaces/Project/darknet-tfl/tfl/MissFreshSmartShelf/backup #模型数据存储位置(废弃)
'''
data_info = 'missfresh.data'