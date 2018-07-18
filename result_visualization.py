from load_data import load_data
import json
import os
import numpy as np
# import win32api
# import win32con

from PIL import Image, ImageDraw, ImageFont


class visualize(load_data):
    """docstring for visualize"""

    def __init__(self, map_file, file_path, pic_path, is_gt=True, threshold=0.05):
        super(visualize, self).__init__(
            file_path=file_path, is_gt=is_gt, threshold=0.05)
        # super class comment
        # self.boxes = {}
        # self.scores = {}
        # self.class_ids = {}
        # self.file_path = file_path
        # self.threshold = threshold
        # self.cat_num = 0
        self.pic_path = pic_path
        self.map = []
        self.image_list = os.listdir(pic_path)
        with open(map_file) as f:
            js_format = json.load(f)
            self.map = js_format['images']

    def find_id(self, file_name):
        for item in self.map:
            if item['file_name'] == file_name:
                return item['id']

    def rand_draw_bbox(self):
        t = np.random.randint(0, len(self.image_list))
        # print(len(self.image_list))
        # print(self.image_list[t])
        # return
        item = self.image_list[t]
        id = self.find_id(item)
        img = Image.open(self.pic_path + item)
        img_d = ImageDraw.Draw(img)
        for i in self.boxes[id]:
            for j in self.boxes[id][i]:
                img_d.line((j[0], j[1], j[0] + j[2], j[1]),
                           fill='red', width=5)
                img_d.line((j[0] + j[2], j[1], j[0] + j[2],
                            j[1] + j[3]), fill='red', width=5)
                img_d.line((j[0] + j[2], j[1] + j[3], j[0],
                            j[1] + j[3]), fill='red', width=5)
                img_d.line((j[0], j[1] + j[3], j[0], j[1]),
                           fill='red', width=5)
        img.show()


if __name__ == "__main__":
    a = visualize(file_path='data/Retinanet_size300/model_iter10499.json',
                  map_file='data/instances_gt_test.json', is_gt=False, pic_path='data/Image_test/')
    a.rand_draw_bbox()
    # im = Image.open('data/Image_test/frames_00007.jpg')
