#coding:utf-8 允许中文注释
# Author: yongyuan.name
#from PIL import Image, ImageEnhance, ImageOps, ImageFile 
import numpy as np
import cv2
import os

def imsize(image, width):
    arr=image.shape
    height=int((arr[0]/arr[1])*width)
    image=cv2.resize(image,(width,height))
    return image


def show(img_names): 
    for img_name in img_names:
        print img_name
    #    name = 'MissFresh_0329_train_0445_004'
    #    image_path = wd + 'flip_image/' + name + '.jpg'
        image_path = os.path.join(dataset_dir,'JPEGImages','%s.jpg'%img_name)
        image= cv2.imread(image_path)
        
    #    box = np.loadtxt(wd + 'txt/' + name + '.txt')
        box = open(os.path.join(wd ,'results\\predict\\missfresh-yolo-voc-800-0507\\yolo-voc-800_24000\\test\\' + img_name + '.txt'))
        boxn = box.readlines()
        box.close()
        arr = image.shape
        
        width = arr[1]
        height = arr[0]        
        
        
        len_box = len(boxn)
        for i in range(0,len_box):
            box2=boxn[i].split()
            if(float(box2[1])<0.65): 
                break
            box2[2]=float(box2[2])
            box2[3]=float(box2[3])
            box2[4]=float(box2[4])
            box2[5]=float(box2[5])
            
            box1 = (width*box2[2]-width*box2[4]/2, height*box2[3]-height*box2[5]/2, width*box2[4], height*box2[5])
        
            array_box1 = np.array(box1)
            array_box1 = array_box1.astype(int)
    #        cv2.rectangle(image,(array_box1[0],array_box1[1]),(array_box1[0]+array_box1[2],array_box1[1]+array_box1[3]),(0,255,0),3,8,0)
    #        cv2.putText(image,('%s'%(int(box2[0])+1)),(int(box1[0])+5,int(box1[1])+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,200,255),thickness=2)
            roii = image[array_box1[1]:array_box1[1]+array_box1[3],array_box1[0]:array_box1[0]+array_box1[2]]
            win_name = box2[0]
            print box2[1]
            cv2.imshow(win_name,roii)
            cv2.moveWindow(win_name,10,10)
            #cv2.imwrite('C:\\Users\\lvsikai\\Desktop\\'+img_name+'_'+str(i)+'.jpg',roii)
            k = cv2.waitKey(0)
            if k==ord('q'):
                cv2.destroyAllWindows()
                break
            elif k==ord('c'):
                cv2.destroyWindow(win_name)
            
    '''     
    image=imsize(image, 600)
    win_name = '%s' % (img_name)
    cv2.imshow(win_name,image)
#    cv2.imwrite(wd + 'results/predict/' +win_name+'_predict.jpg',image)
    
    cv2.moveWindow(win_name,10,10)
    #cv2.imwrite('D:/Text_image/Goods_id/cut_result0326/'+name+'.jpg',image)
    k = cv2.waitKey(0)
    if k==ord('q'):
        cv2.destroyAllWindows()
        break
    elif k==ord('c'):
        cv2.destroyWindow(win_name)
    '''
    
def save_imgs(img_names,dataset_dir,wd):
        for img_name in img_names:
            print img_name
            image_path = os.path.join(dataset_dir,'JPEGImages','%s.jpg'%img_name)
            image= cv2.imread(image_path)
            box = open(os.path.join(dataset_dir ,'dk_labels' , img_name + '.txt'))
            boxn = box.readlines()
            box.close()
            arr = image.shape
            
            width = arr[1]
            height = arr[0]        
            
            
            len_box = len(boxn)
            for i in range(0,len_box):
                box2=boxn[i].split()
                box2[1]=min([max([0,float(box2[1])]),1])
                box2[2]=min([max([0,float(box2[2])]),1])
                box2[3]=min([max([0,float(box2[3])]),1])
                box2[4]=min([max([0,float(box2[4])]),1])
                
                box1 = (max([0,width*box2[1]-width*box2[3]/2]), max([0,height*box2[2]-height*box2[4]/2]), min([width,width*box2[3]]), min([height,height*box2[4]]))
            
                array_box1 = np.array(box1)
                array_box1 = array_box1.astype(int)
                roii = image[array_box1[1]:array_box1[1]+array_box1[3],array_box1[0]:array_box1[0]+array_box1[2]]
                if not os.path.exists(wd + '\\'+box2[0]):
                    os.mkdir(wd + '\\'+box2[0])
                cv2.imwrite(wd + '\\'+box2[0]+'\\'+img_name+'_'+str(i)+'.jpg',roii)
                
if __name__ == "__main__":
    #wd = 'C:\\Users\\lvsikai\\Desktop\\repositories\\eval_yolo_detection'
    wd = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\missfresh_retrieval_lib'
    dataset_dir = 'C:\\Users\\lvsikai\\Desktop\\Yi+\\Missfresh\\7MissFreshSmartShelf_Exp'
    dataset_file = os.path.join(dataset_dir, 'ImageSets', 'Main', 'val.txt')
    f = open(dataset_file)
    img_names = f.readlines()
    f.close()
    img_names = [x.strip() for x in img_names]
    #DataSets = [('yolo-voc-800_24000', 'test', 'missfresh-yolo-voc-800-0507')]
    save_imgs(img_names,dataset_dir,wd)
      












