import run_eval as Evaluation
import os
import operator
import matplotlib.pyplot as plt


class accmulation:
    """docstring for accmulation"""
    # do pre_ckp evaluation and draw mAP curve for modules

    def __init__(self, gt_file, pred_path, cat=True, threshold=0.05, iou_threshold=0.5):
        super(accmulation, self).__init__()
        self.gt_file = gt_file
        self.pred_file = os.listdir(pred_path)
        self.pred_file = [os.path.join(pred_path, j) for j in self.pred_file]
        self.cat = cat
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.eval = {'mAP': {}, 'mAP50': {}, 'mAP75': {}}

    def load_mAP(self):
        for item in self.pred_file:
            cpk_id = int(item.split('_')[-1][4:-5]) + 1
            modules_tmp = Evaluation.evaluation(self.gt_file,
                                                item, threshold=self.threshold, cat=self.cat, iou_threshold=self.iou_threshold)
            modules_tmp.get_mAP()
            self.eval['mAP'][cpk_id] = modules_tmp.eval['mAP']
            if self.cat:
                self.eval['mAP50'][cpk_id] = modules_tmp.eval['mAP50']
                self.eval['mAP75'][cpk_id] = modules_tmp.eval['mAP75']
        self.eval['mAP'] = (sorted(
            self.eval['mAP'].items(), key=operator.itemgetter(0)))
        if self.cat:
            self.eval['mAP50'] = (sorted(
                self.eval['mAP50'].items(), key=operator.itemgetter(0)))
            self.eval['mAP75'] = (sorted(
                self.eval['mAP75'].items(), key=operator.itemgetter(0)))
        return

    def draw_map_curve(self):
        self.load_mAP()
        x = [self.eval['mAP'][i][0] for i in range(len(self.eval['mAP']))]
        y = [self.eval['mAP'][i][1] for i in range(len(self.eval['mAP']))]

        plt.plot(x, y, label='mAP', color='r', linewidth=2,
                 marker='o', markersize=6)
        if self.cat:
            x1 = [self.eval['mAP50'][i][0]
                  for i in range(len(self.eval['mAP50']))]
            y1 = [self.eval['mAP50'][i][1]
                  for i in range(len(self.eval['mAP50']))]
            plt.plot(x1, y1, label='mAP50', color='b', linewidth=2,
                     marker='o', markersize=6)
            x2 = [self.eval['mAP75'][i][0]
                  for i in range(len(self.eval['mAP75']))]
            y2 = [self.eval['mAP75'][i][1]
                  for i in range(len(self.eval['mAP75']))]
            plt.plot(x2, y2, label='mAP75', color='y', linewidth=2,
                     marker='o', markersize=6)
        plt.xlabel('#Iterations')
        plt.ylabel('mAP')
        plt.title('mAP curve ')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # ckp must be saved as ***_5000.json
    a = accmulation('data/instances_gt_test.json',
                    './data/Retinanet_size300', threshold=0.5, cat=True)
    a.draw_map_curve()
