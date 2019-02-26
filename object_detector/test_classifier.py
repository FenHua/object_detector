import argparse as ap
import os
import cv2
import matplotlib.pyplot as plt
import scipy.misc
from skimage.feature import hog
from sklearn.externals import joblib
from config import *
from utils import sliding_window, pyramid, non_max_suppression, rgb2gray

class Detector:
    def __init__(self, downscale=1.5, window_size=(40,40), window_step_size=10, threshold=0.4):
        self.clf = joblib.load(MODEL_PATH)
        self.downscale = downscale
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.threshold = threshold

    def detect(self, image):
        clone = image.copy()
        image = rgb2gray(image)
        detections = []  # 记录识别的目标
        downscale_power = 0  # 当前下采样系数
        # 迭代下采样
        for im_scaled in pyramid(image, downscale=self.downscale, min_size=self.window_size):
            if im_scaled.shape[0] < self.window_size[1] or im_scaled.shape[1] < self.window_size[0]:
                # 如果采样尺度小于模板窗，就停止迭代
                break
            for (x, y, im_window) in sliding_window(im_scaled, self.window_step_size,self.window_size):
                if im_window.shape[0] != self.window_size[1] or im_window.shape[1] != self.window_size[0]:
                    continue
                feature_vector = hog(im_window,block_norm="L1")  # 计算HOG特征
                X = np.array([feature_vector])
                prediction = self.clf.predict(X)
                if prediction == 1:
                    x1 = int(x * (self.downscale ** downscale_power))
                    y1 = int(y * (self.downscale ** downscale_power))
                    detections.append((x1, y1,
                                       x1 + int(self.window_size[0] * (self.downscale ** downscale_power)),
                                       y1 + int(self.window_size[1] * (self.downscale ** downscale_power))))
            downscale_power += 1  # 移动到下一个尺度
        clone_before_nms = clone.copy()  # 用来显示NMS处理前的结果
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(clone_before_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  # 描框
        detections = non_max_suppression(np.array(detections), self.threshold)  # NMS处理后的结果
        clone_after_nms = clone
        # NMS处理后的结果
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(clone_after_nms, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  # 描框
        return clone_before_nms, clone_after_nms

if __name__ == '__main__':

    parser = ap.ArgumentParser()  # 命令行解析
    # parser.add_argument('-i', '--images_dir_path', help='Path to the test images dir',required=True)
    parser.add_argument('-v', '--visualize', help='Visualize the sliding window',action='store_true')
    args = vars(parser.parse_args())
    visualize_det = args['visualize']
    # image_dir_path = args['images_dir_path']
    image_dir_path="/home/yhq/Desktop/object_detector/data/datasets/test/"
    detector = Detector(downscale=PYRAMID_DOWNSCALE, window_size=WINDOW_SIZE,
                        window_step_size=WINDOW_STEP_SIZE, threshold=THRESHOLD)
    for image_name in os.listdir(image_dir_path):
        if image_name == '.DS_Store':
            continue
        image = scipy.misc.imread(os.path.join(image_dir_path, image_name))  # 读图
        image_before_nms, image_after_nms = detector.detect(image)  # 识别结果
        # 显示NMS前的结果
        plt.imshow(image_before_nms)
        plt.xticks([]), plt.yticks([])
        plt.show()
        # 显示NMS后的结果
        plt.imshow(image_after_nms)
        plt.xticks([]), plt.yticks([])
        plt.show()