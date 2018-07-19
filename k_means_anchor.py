# coding=utf-8
# k-means ++ for YOLOv2 anchors
# 通过k-means ++ 算法获取YOLOv2需要的anchors的尺寸
import numpy as np
from load_data import load_data


class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

# 计算两个box在某个轴上的重叠部分
# x1是box1的中心在该轴上的坐标
# len1是box1在该轴上的长度
# x2是box2的中心在该轴上的坐标
# len2是box2在该轴上的长度
# 返回值是该轴上重叠的长度


def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


# 计算box a 和box b 的交集面积
# a和b都是Box类型实例
# 返回值area是box a 和box b 的交集面积
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


# 计算 box a 和 box b 的并集面积
# a和b都是Box类型实例
# 返回值u是box a 和box b 的并集面积
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


# 计算 box a 和 box b 的 iou
# a和b都是Box类型实例
# 返回值是box a 和box b 的iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


class k_means_anchor(load_data):
    """docstring for k-means"""

    def __init__(self, file_path, n_anchors):
        super(k_means_anchor, self).__init__(
            file_path=file_path, is_gt=True)
        # self.boxes = {}
        # self.scores = {}
        # self.class_ids = {}
        # self.file_path = file_path
        # self.threshold = threshold
        # self.cat_num = 0
        self.n_anchors = n_anchors
        self.k_means_boxes = []
        self.centroids = []
        for i in range(len(self.boxes)):
            # print(self.boxes[i + 1])
            for j in self.boxes[i + 1]:
                for item in self.boxes[i + 1][j]:
                    self.k_means_boxes.append(
                        Box(0, 0, item[2], item[3]))
        # print(len(self.k_means_boxes))

# 使用k-means ++ 初始化 centroids，减少随机初始化的centroids对最终结果的影响
# boxes是所有bounding boxes的Box对象列表
# n_anchors是k-means的k值
# 返回值centroids 是初始化的n_anchors个centroid
    def init_centroids(self):
        centroids = []
        boxes_num = len(self.k_means_boxes)
        centroid_index = np.random.choice(boxes_num, 1)
        centroids.append(self.k_means_boxes[centroid_index[0]])

        print(centroids[0].w, centroids[0].h)

        for centroid_index in range(0, self.n_anchors - 1):

            sum_distance = 0
            distance_thresh = 0
            distance_list = []
            cur_sum = 0

            for box in self.k_means_boxes:
                min_distance = 1
                for centroid_i, centroid in enumerate(centroids):
                    distance = (1 - box_iou(box, centroid))
                    if distance < min_distance:
                        min_distance = distance
                sum_distance += min_distance
                distance_list.append(min_distance)

            distance_thresh = sum_distance * np.random.random()

            for i in range(0, boxes_num):
                cur_sum += distance_list[i]
                if cur_sum > distance_thresh:
                    centroids.append(self.k_means_boxes[i])
                    print(self.k_means_boxes[i].w, self.k_means_boxes[i].h)
                    break

        self.centroids = centroids

    # 进行 k-means 计算新的centroids
    # boxes是所有bounding boxes的Box对象列表
    # n_anchors是k-means的k值
    # centroids是所有簇的中心
    # 返回值new_centroids 是计算出的新簇中心
    # 返回值groups是n_anchors个簇包含的boxes的列表
    # 返回值loss是所有box距离所属的最近的centroid的距离的和
    def do_kmeans(self):
        loss = 0
        groups = []
        new_centroids = []
        for i in range(self.n_anchors):
            groups.append([])
            new_centroids.append(Box(0, 0, 0, 0))

        for box in self.k_means_boxes:
            min_distance = 1
            group_index = 0
            for centroid_index, centroid in enumerate(self.centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
                    group_index = centroid_index
            groups[group_index].append(box)
            loss += min_distance
            new_centroids[group_index].w += box.w
            new_centroids[group_index].h += box.h

        for i in range(self.n_anchors):
            new_centroids[i].w /= len(groups[i])
            new_centroids[i].h /= len(groups[i])
        self.centroids = new_centroids

        return new_centroids, groups, loss

    # 计算给定bounding boxes的n_anchors数量的centroids
    # label_path是训练集列表文件地址
    # n_anchors 是anchors的数量
    # loss_convergence是允许的loss的最小变化值
    # grid_size * grid_size 是栅格数量
    # iterations_num是最大迭代次数
    # plus = 1时启用k means ++ 初始化centroids
    def compute_centroids(self, loss_convergence, grid_size, iterations_num, plus):

        if plus:
            self.init_centroids(self.k_means_boxes, self.n_anchors)
        else:
            centroid_indices = np.random.choice(
                len(self.k_means_boxes), self.n_anchors)
            self.centroids = []
            for centroid_index in centroid_indices:
                self.centroids.append(self.k_means_boxes[centroid_index])

        # iterate k-means
        self.centroids, groups, old_loss = self.do_kmeans()
        iterations = 1
        while (True):
            self.centroids, groups, loss = self.do_kmeans()
            iterations = iterations + 1
            print("loss = %f" % loss)
            if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
                break
            old_loss = loss

            for centroid in self.centroids:
                print(centroid.w * grid_size, centroid.h * grid_size)

        # print result
        for centroid in self.centroids:
            print("k-means result：\n")
            print(centroid.w * grid_size, centroid.h * grid_size)


if __name__ == '__main__':
    a = k_means_anchor('data/instances_gt_test.json', 3)

    loss_convergence = 1e-6
    grid_size = 1
    iterations_num = 100
    plus = 0

    a.compute_centroids(loss_convergence,
                        grid_size, iterations_num, plus)
