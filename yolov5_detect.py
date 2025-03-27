# 导入os模块，用于文件和目录操作
import os
# 导入urllib模块，用于URL操作
import urllib
# 导入traceback模块，用于异常跟踪
import traceback
# 导入time模块，用于时间相关的操作
import time
# 导入sys模块，用于与Python解释器交互
import sys
# 导入numpy模块，用于数值计算
import numpy as np
# 导入cv2模块，用于图像处理
import cv2
# 导入RKNN模块，用于RKNN模型的创建和推理
from rknn.api import RKNN

# 定义ONNX模型的路径
ONNX_MODEL = 'yolov5s.onnx'
# 定义导出的RKNN模型的路径
RKNN_MODEL = 'yolov5s_quant.rknn'
# 定义输入图像的路径
IMG_PATH = './street.jpg'
# 定义数据集文件的路径，用于量化
DATASET = './datasets.txt'

# 定义是否开启量化
QUANTIZE_ON = True

# 定义目标检测阈值
OBJ_THRESH = 0.25
# 定义非极大值抑制阈值
NMS_THRESH = 0.45
# 定义输入图像的尺寸
IMG_SIZE = 640

# 定义类别列表
CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

# 定义sigmoid函数，用于计算激活值
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 将边界框格式从[x, y, w, h]转换为[x1, y1, x2, y2]
def xywh2xyxy(x):
    # 复制输入数组
    y = np.copy(x)
    # 计算左上角x坐标
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    # 计算左上角y坐标
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    # 计算右下角x坐标
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    # 计算右下角y坐标
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# 处理模型输出，计算边界框、置信度和类别概率
def process(input, mask, anchors):
    # 根据mask选择anchors
    anchors = [anchors[i] for i in mask]
    # 获取特征图的高度和宽度
    grid_h, grid_w = map(int, input.shape[0:2])

    # 计算边界框置信度
    box_confidence = sigmoid(input[..., 4])
    # 扩展维度
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    # 计算类别概率
    box_class_probs = sigmoid(input[..., 5:])

    # 计算边界框中心点坐标
    box_xy = sigmoid(input[..., :2])*2 - 0.5

    # 生成网格点
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    # 计算边界框宽高
    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    # 合并边界框中心点坐标和宽高
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

# 过滤边界框，只保留置信度和类别概率大于阈值的边界框
def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    # 重塑边界框数组
    boxes = boxes.reshape(-1, 4)
    # 重塑置信度数组
    box_confidences = box_confidences.reshape(-1)
    # 重塑类别概率数组
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    # 获取置信度大于阈值的索引
    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    # 过滤边界框
    boxes = boxes[_box_pos]
    # 过滤置信度
    box_confidences = box_confidences[_box_pos]
    # 过滤类别概率
    box_class_probs = box_class_probs[_box_pos]

    # 获取类别最大概率
    class_max_score = np.max(box_class_probs, axis=-1)
    # 获取类别索引
    classes = np.argmax(box_class_probs, axis=-1)
    # 获取类别最大概率大于阈值的索引
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    # 过滤边界框
    boxes = boxes[_class_pos]
    # 过滤类别
    classes = classes[_class_pos]
    # 计算得分
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores

# 非极大值抑制，去除重叠的边界框
def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    # 获取边界框的坐标
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    # 计算边界框的面积
    areas = w * h
    # 按得分降序排序
    order = scores.argsort()[::-1]

    keep = []
    # 遍历排序后的边界框
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

# 后处理函数，对模型输出进行处理，得到最终的边界框、类别和得分
def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    # 遍历每个输出
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    # 合并所有边界框
    boxes = np.concatenate(boxes)
    # 转换边界框格式
    boxes = xywh2xyxy(boxes)
    # 合并所有类别
    classes = np.concatenate(classes)
    # 合并所有得分
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    # 遍历每个类别
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    # 合并所有边界框
    boxes = np.concatenate(nboxes)
    # 合并所有类别
    classes = np.concatenate(nclasses)
    # 合并所有得分
    scores = np.concatenate(nscores)

    return boxes, classes, scores

# 在图像上绘制边界框、类别和得分
def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    # 遍历每个边界框
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # 打印类别和得分
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        # 打印边界框坐标
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        # 绘制边界框
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        # 绘制类别和得分
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

# 对图像进行letterbox处理，使其符合模型输入尺寸
def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # 获取图像的高度和宽度
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 计算填充比例
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

# 主函数
if __name__ == '__main__':

    # 创建RKNN对象
    rknn = RKNN(verbose=True)

    # 配置模型
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # 加载ONNX模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 构建模型
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 导出RKNN模型
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 初始化运行时环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 读取输入图像
    img = cv2.imread(IMG_PATH)
    # 将图像转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 调整图像大小
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 运行模型
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # 保存模型输出
    np.save('./onnx_yolov5_0.npy', outputs[0])
    np.save('./onnx_yolov5_1.npy', outputs[1])
    np.save('./onnx_yolov5_2.npy', outputs[2])
    print('done')

    # 后处理
    input0_data = outputs[0]
    input1_data = outputs[1]
    input2_data = outputs[2]

    input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = yolov5_post_process(input_data)

    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        draw(img_1, boxes, scores, classes)
        cv2.imwrite('result.png', img_1)

    rknn.release()