# 设置变量参数
import configparser as cp
import json
import random
import numpy as np

config = cp.RawConfigParser()
config.read('/home/yhq/Desktop/object_detector/data/config/config.cfg')  # 配置文件路经
WINDOW_SIZE = json.loads(config.get('hog', 'window_size'))  # json.loads将json对象转化python对象，dumps方法将python对象转化为json格式
WINDOW_STEP_SIZE = config.getint('hog', 'window_step_size')  # 获取指定key为hog.window_step_size的value
ORIENTATIONS = config.getint('hog', 'orientations')
PIXELS_PER_CELL = json.loads(config.get('hog', 'pixels_per_cell'))
CELLS_PER_BLOCK = json.loads(config.get('hog', 'cells_per_block'))
VISUALISE = config.getboolean('hog', 'visualise')
NORMALISE = config.get('hog', 'normalise') # 返回字符串
if NORMALISE == 'None':
    NORMALISE = None
THRESHOLD = config.getfloat('nms', 'threshold')
MODEL_PATH = config.get('paths', 'model_path')
PYRAMID_DOWNSCALE = config.getfloat('general', 'pyramid_downscale')
POS_SAMPLES = config.getint('general', 'pos_samples')
NEG_SAMPLES = config.getint('general', 'neg_samples')
RANDOM_STATE = 31 # 随机种子
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)