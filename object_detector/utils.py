import os
import cv2
import numpy as np
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', 'pgm')


def rgb2gray(rgb_image):
    return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.144])# 将RGB转换为灰度


def pyramid(image, downscale=1.5, min_size=(30, 30)):
    yield image  # 原图
    while True:
        # 图像金字塔
        w = int(image.shape[1] / downscale)  # 计算出新的图片大小
        image = resize(image, width=w)
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            # 如果resize的图不满足最小尺寸要求就停止
            break
        yield image  # 产生金字塔中的下一个图片


def sliding_window(image, step_size, window_size):
    # 函数返回输入图片的一系列切片和切片在原图中的位置坐标
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # 窗口以step_size大小移动
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])  # 记录当前的窗口


def non_max_suppression(boxes, overlap_thresh=0.5):
    # 非极大值抑制NMS
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")  # 将数据类型转换为float型
    pick = []  # 初始化筛选的结果
    # 获取bounding box坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算每个box面积
    idxs = np.argsort(y2)  # 从小到大排序
    # NMS
    while len(idxs) > 0:
        # 记录最后一个index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)  # 获取最大boundingbox的坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # 计算重叠区域的长和宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]  # 计算重叠率
        # 删除重叠率大于阈值的box
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype("int")  # 返回box坐标信息


def bb_intersection(box_a, box_b):
    # 计算俩box相交的面积
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    t1 = xB - xA + 1
    t2 = yB - yA + 1
    if t1 <= 0 or t2 <= 0:
        intersection_area = 0
    else:
        intersection_area = (xB - xA + 1) * (yB - yA + 1)
    return intersection_area


def bb_intersection_over_union(box_a, box_b):
    intersection_area = bb_intersection(box_a, box_b)
    # 计算预测box和真实box的面积
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = intersection_area / (box_a_area + box_b_area - intersection_area)# 计算IOU
    return iou


def is_image_file(file_name):
    ext = file_name[file_name.rfind('.'):].lower()  # 返回文件名的后缀(小写)
    return ext in IMAGE_EXTENSIONS


def list_images(base_path, contains=None):
    # 返回有效的文件路经名
    return list_files(base_path, valid_exts=IMAGE_EXTENSIONS, contains=contains)


def list_files(base_path, valid_exts=IMAGE_EXTENSIONS, contains=None):
    # 循环目录结构
    for (root_cir, dir_names, file_names) in os.walk(base_path):
        # os.walk方法通过在目录树中游走输出在目录中的文件名
        for file_name in file_names:
            if contains is not None and file_name.find(contains) == -1:
                continue  # 忽略无效的文件
            ext = file_name[file_name.rfind('.'):].lower()# 当前文件的数据类型
            if ext.endswith(valid_exts):
                # 检查当前文件的文件类型
                image_path = os.path.join(root_cir, file_name).replace(" ", "\\ ")  # 建立文件的路经
                yield image_path


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    # 仅给长宽一个指标时
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resize图片
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
