import argparse as ap
import os
import scipy.misc
from skimage.feature import hog
from sklearn.externals import joblib
from tqdm import tqdm
from config import *
from utils import rgb2gray

def extract_features(image_dir_path, feature_dir_path, n_samples, ext='.feat'):
    # 计算训练样本的hog特征
    progress_bar = tqdm(total=n_samples) # tqdm是python进度条
    i = 0
    # os.listdir 方法用于返回指定文件夹包含的文件或文件夹的名字的列表
    for image_path in os.listdir(image_dir_path):
        if i == n_samples:
            break
        image = scipy.misc.imread(os.path.join(image_dir_path, image_path))
        # image = rgb2gray(image)
        features = hog(image, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL,cells_per_block=CELLS_PER_BLOCK)
        features_file_name = image_path.split('.')[0] + ext  # 获取除去后缀的文件名字
        features_dir_path = feature_dir_path
        features_file_path = os.path.join(features_dir_path, features_file_name) #路经拼接
        joblib.dump(features, features_file_path, compress=3)  # 将特征进行保存，压缩比为3
        i += 1
        progress_bar.update(1)

if __name__ == '__main__':
    '''
    parser = ap.ArgumentParser()  # 命令行解析
    parser.add_argument('-pi', '--pos_image_dir_path', help='Path to pos images',required=True)
    parser.add_argument('-ni', '--neg_image_dir_path', help='Path to neg images',required=True)
    parser.add_argument('-pf', '--pos_features_path', help='Path to the positive features directory',required=True)
    parser.add_argument('-nf', '--neg_features_path', help='Path to the negative features directory',required=True)
    args = vars(parser.parse_args())
    pos_image_dir_path = args['pos_image_dir_path']
    neg_image_dir_path = args['neg_image_dir_path']
    pos_features_path = args['pos_features_path']
    neg_features_path = args['neg_features_path']
    '''
    pos_image_dir_path = "/home/yhq/Desktop/object_detector/data/datasets/pos_image"
    neg_image_dir_path = "/home/yhq/Desktop/object_detector/data/datasets/neg_image"
    pos_features_path = "/home/yhq/Desktop/object_detector/data/datasets/pos_features"
    neg_features_path = "/home/yhq/Desktop/object_detector/data/datasets/neg_features"
    if not os.path.exists(pos_features_path):
        os.makedirs(pos_features_path)  # 正样本特征目录
    if not os.path.exists(neg_features_path):
        os.makedirs(neg_features_path)  # 负样本特征目录
    print('Calculating descriptors for the training samples and saving them\n')
    print('\nPositive samples extracting ...')
    extract_features(image_dir_path=pos_image_dir_path, feature_dir_path=pos_features_path, n_samples=POS_SAMPLES)
    print('\nNegative samples extracting ...')
    extract_features(image_dir_path=neg_image_dir_path, feature_dir_path=neg_features_path, n_samples=NEG_SAMPLES)
    print('Completed calculating features from training images')