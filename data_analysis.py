from load_data import load_data
from sklearn.cluster import KMeans
import numpy as np
from prettytable import PrettyTable
label_map = 'label_map.txt'


class analysis(load_data):
    """docstring for analysis"""

    def __init__(self, file_path, classes, image_size=(1080, 1920), is_gt=True, threshold=0.05):
        super(analysis, self).__init__(file_path, is_gt=is_gt, threshold=0.05)
        self.image_size = image_size
        self.k_means_center = []
        # self.boxes = {}
        # self.scores = {}
        # self.class_ids = {}
        # self.file_path = file_path
        # self.threshold = threshold
        # self.cat_num = 0
        self.image_nums = np.zeros(
            classes, int)
        self.classes = classes
        self.bbox_nums = np.zeros(classes, int)  # bbox_num of each categories
        self.bbox_width_averaged = np.zeros(
            classes, float)  # bbox_size of each categories
        self.bbox_height_averaged = np.zeros(
            classes, float)  # bbox_size of each categories
        self.multiclass_num = 0
        self.classes_name = []
        self.image_categories = {}
        self.multi_class_image = np.zeros(classes, int)
        f = open(label_map)
        for line in f.readlines():
            lab = line.rstrip().split(' ')
            self.classes_name.append(lab[1])
            self.image_categories[int(lab[0])] = []

    def init_Analysis(self):
        bbox = []
        for i in range(len(self.boxes)):
            # print(a.boxes[i + 1])
            flag = 0
            for j in range(self.cat_num):
                if j + 1 in self.boxes[i + 1]:

                    if flag == 1:
                        self.multiclass_num += 1
                        flag = -1
                    if flag == 0:
                        flag = 1
                    self.image_nums[j] += 1
                    for k in self.boxes[i + 1][j + 1]:
                        self.bbox_nums[j] += 1
                        self.bbox_height_averaged[j] += k[3]
                        self.bbox_width_averaged[j] += k[2]
                        bbox.append([k[2], k[3]])
            if flag == -1:
                for j in range(self.cat_num):
                    if j + 1 in self.boxes[i + 1]:
                        self.multi_class_image[j] += 1
        bbox = np.array(bbox)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(bbox)
        for j in range(self.classes):
            self.bbox_width_averaged[j] /= self.bbox_nums[j]
            self.bbox_height_averaged[j] /= self.bbox_nums[j]
        self.k_means_center = (kmeans.cluster_centers_)

    def show_distribution(self):
        table = PrettyTable(['Category'] + self.classes_name)
        table.align = 'l'
        str_class_num = [str(ite) for ite in self.image_nums]
        table.add_row(['#Images'] + str_class_num)
        str_bbox_num = [str(ite) for ite in self.bbox_nums]
        table.add_row(['#Bbox'] + str_bbox_num)
        str_bbox_width = [str(int(ite)) for ite in self.bbox_width_averaged]
        table.add_row(['#Aver_width'] + str_bbox_width)
        str_bbox_height = [str(int(ite)) for ite in self.bbox_height_averaged]
        table.add_row(['#Aver_height'] + str_bbox_height)
        str_multi_class = [str(ite) for ite in self.multi_class_image]
        table.add_row(['#multiclass'] + str_multi_class)
        print(str(table))
        print('Number of images which have multi-categories:',
              self.multiclass_num)


if __name__ == '__main__':
    # a = load_data('data/instances_gt_test.json')
    # k_means_center = k_means(a)
    # print(k_means_center)
    a = analysis(file_path='data/instances_gt_test.json', classes=7)
    a.init_Analysis()
    a.show_distribution()
    print(a.k_means_center)
