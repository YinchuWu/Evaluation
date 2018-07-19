import run_eval as Evaluation
import os
import operator
import matplotlib.pyplot as plt


class compare_diff:
    """docstring for accmulation"""
    # do pre_ckp evaluation and draw mAP curve for modules

    def __init__(self, gt_file1, pred_path1, gt_file2, pred_path2, cat=True, threshold=0.05, iou_threshold=0.5):
        super(compare_diff, self).__init__()
        self.gt_file1 = gt_file1
        self.pred_file1 = os.listdir(pred_path1)
        self.pred_file1 = [os.path.join(pred_path1, j)
                           for j in self.pred_file1]

        self.gt_file2 = gt_file2
        self.pred_file2 = os.listdir(pred_path2)
        self.pred_file2 = [os.path.join(pred_path2, j)
                           for j in self.pred_file2]

        self.cat = cat
        self.threshold = threshold
        self.iou_threshold = iou_threshold

        self.eval1 = {'mAP': {}, 'mAP50': {}, 'mAP75': {}}
        self.eval2 = {'mAP': {}, 'mAP50': {}, 'mAP75': {}}

    def load_mAP(self):
        for item in self.pred_file1:
            cpk_id = int(item.split('_')[-1][4:-5]) + 1
            modules_tmp = Evaluation.evaluation(self.gt_file1,
                                                item, threshold=self.threshold, cat=self.cat, iou_threshold=self.iou_threshold)
            modules_tmp.get_mAP()
            self.eval1['mAP'][cpk_id] = modules_tmp.eval['mAP']
            if self.cat:
                self.eval1['mAP50'][cpk_id] = modules_tmp.eval['mAP50']
                self.eval1['mAP75'][cpk_id] = modules_tmp.eval['mAP75']
        self.eval1['mAP'] = (sorted(
            self.eval1['mAP'].items(), key=operator.itemgetter(0)))
        if self.cat:
            self.eval1['mAP50'] = (sorted(
                self.eval1['mAP50'].items(), key=operator.itemgetter(0)))
            self.eval1['mAP75'] = (sorted(
                self.eval1['mAP75'].items(), key=operator.itemgetter(0)))

        for item in self.pred_file2:
            cpk_id = int(item.split('_')[-1][4:-5]) + 1
            modules_tmp = Evaluation.evaluation(self.gt_file2,
                                                item, threshold=self.threshold, cat=self.cat, iou_threshold=self.iou_threshold)
            modules_tmp.get_mAP()
            self.eval2['mAP'][cpk_id] = modules_tmp.eval['mAP']
            if self.cat:
                self.eval2['mAP50'][cpk_id] = modules_tmp.eval['mAP50']
                self.eval2['mAP75'][cpk_id] = modules_tmp.eval['mAP75']
        self.eval2['mAP'] = (sorted(
            self.eval2['mAP'].items(), key=operator.itemgetter(0)))
        if self.cat:
            self.eval2['mAP50'] = (sorted(
                self.eval2['mAP50'].items(), key=operator.itemgetter(0)))
            self.eval2['mAP75'] = (sorted(
                self.eval2['mAP75'].items(), key=operator.itemgetter(0)))

        return

    def draw_map_curve(self):
        self.load_mAP()
        x1 = [self.eval1['mAP'][i][0] for i in range(len(self.eval1['mAP']))]
        y1 = [self.eval1['mAP'][i][1] for i in range(len(self.eval1['mAP']))]
        x2 = [self.eval2['mAP'][i][0] for i in range(len(self.eval2['mAP']))]
        y2 = [self.eval2['mAP'][i][1] for i in range(len(self.eval2['mAP']))]

        plt.plot(x1, y1, label='mAP1', color='r', linewidth=2,
                 marker='o', markersize=6)
        plt.plot(x2, y2, label='mAP2', color='g', linewidth=2,
                 marker='o', markersize=6)

        plt.xlabel('#Iterations')
        plt.ylabel('mAP')
        plt.title('mAP curve ')
        plt.legend()
        plt.show()
        plt.savefig('compare_diff.png')


if __name__ == '__main__':
    # ckp must be saved as ***_5000.json
    a = compare_diff('data/instances_gt_test.json',
                     './data/Modified_anchor', 'data/instances_gt_test.json',
                     './data/Retinanet_size300', threshold=0.05, cat=True)
    a.draw_map_curve()
