# 导入所需的Python库
import os  # 操作系统接口，用于文件和路径操作
import urllib  # 用于处理URL和网络请求
import traceback  # 用于异常追踪和调试
import time  # 时间相关的函数
import sys  # Python系统相关的功能
import numpy as np  # 科学计算库，提供多维数组支持
import cv2  # OpenCV库，用于图像处理
from rknn.api import RKNN  # RKNN API，用于模型转换和推理
from math import exp  # 数学库中的指数函数

# 模型和数据集相关的配置
ONNX_MODEL = './weights/yolov8s.dict.onnx'  # ONNX模型文件路径
RKNN_MODEL = './weights/yolov8s.int.rknn'  # 转换后的RKNN模型保存路径
DATASET = './detect_datasets.txt'  # 数据集文件路径

QUANTIZE_ON = True  # 是否开启量化，True表示开启量化以减少模型大小和加快推理速度
QUANTIZE_ON = False  # 是否开启量化，True表示开启量化以减少模型大小和加快推理速度

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# CLASSES = ['pedestrians','riders','partially-visible-person','ignore-regions','crowd']

# 模型推理相关的全局变量
meshgrid = []  # 用于存储特征图的网格坐标

class_num = len(CLASSES)  # 类别数量
headNum = 3  # YOLOv8的检测头数量
strides = [8, 16, 32]  # 三个检测头的步长，用于计算特征图的缩放比例
mapSize = [[80, 80], [40, 40], [20, 20]]  # 三个检测头的特征图大小
nmsThresh = 0.5  # NMS阈值，用于去除重叠框
objectThresh = 0.2  # 目标置信度阈值，用于过滤低置信度的检测框

input_imgH = 640  # 输入图像的高度
input_imgW = 640  # 输入图像的宽度
"""
- bbox积分形式转换为4 d bbox格式
对Head输出的bbox分支进行转换，利用Softmax和Conv计算将积分形式转换为4维bbox格式
- 维度变换
YOLOv8输出特征图尺度为80x80、40x40和20x20的三个特征图。Head部分输出分类和回归共6个尺
度的特征图。
将3个不同尺度的类别预测分支、bbox预测分支进行拼接，并进行维度变换。为了后续方便处理，会将
原先的通道维度置换到最后，类别预测分支和bbox预测分支shape分别为
（b,80x80+40x40+20x20,80)=(b,8400,80),(b,8400,4)

- 解码还原到原图尺度
分类预测分支进行Sigmoid计算，而bbox预测分支需要进行解码，还原为真实的原图解码后xyxy格式。
- NMS筛选最终的检测框


也就是将原始合并的输出，拆分成6个输出(3对分支：类别、框回归)，主要有以下两个考虑：
- 原始的bounding box解码包含在onnx中，使用NPU推理效率不高，拆除decode,换成在CPU执行
- 类别输出和框输出的数值范围不一致，量化可能导致精度下降
    框回归的average,min,max:3.151631、0.09048931、11.446398
    类别的average,min,max:-15.843855、-21.935265、-2.0644956


git clone https://github.com/ultralytics/ultralytics.git
git checkout a05edfbc27d74e6dce9d0f169036282042aadb9
git apply  enpei.modify.patch
"""

# 定义检测框类，用于存储单个目标的检测结果
class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, head):
        self.classId = classId  # 目标类别ID
        self.score = score      # 置信度分数
        self.xmin = xmin        # 边界框左上角x坐标
        self.ymin = ymin        # 边界框左上角y坐标
        self.xmax = xmax        # 边界框右下角x坐标
        self.ymax = ymax        # 边界框右下角y坐标
        self.head = head        # 来自哪个检测头的预测结果

# 生成特征图的网格坐标
def GenerateMeshgrid():
    # 遍历每个检测头
    for index in range(headNum):  # 12800 16000 16800
        # 遍历特征图的每个网格点
        for i in range(mapSize[index][0]):  # 遍历高度
            for j in range(mapSize[index][1]):  # 遍历宽度
                # 将网格点的x、y坐标添加到meshgrid列表中
                # 加0.5是为了将坐标对齐到网格中心
                meshgrid.append(j + 0.5)  # x坐标
                meshgrid.append(i + 0.5)  # y坐标


# 计算两个边界框的交并比(IOU)
def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    # 计算两个框的交集区域的坐标
    xmin = max(xmin1, xmin2)  # 交集区域的左上角x坐标
    ymin = max(ymin1, ymin2)  # 交集区域的左上角y坐标
    xmax = min(xmax1, xmax2)  # 交集区域的右下角x坐标
    ymax = min(ymax1, ymax2)  # 交集区域的右下角y坐标
    # 计算交集区域的宽度和高度
    innerWidth = xmax - xmin
    innerHeight = ymax - ymin
    # 确保宽度和高度不为负数
    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0
    # 计算交集面积
    innerArea = innerWidth * innerHeight
    # 计算两个框的面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # 第一个框的面积
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # 第二个框的面积
    # 计算并集面积
    total = area1 + area2 - innerArea
    # 返回IOU值（交集面积除以并集面积）
    return innerArea / total


# 非极大值抑制(NMS)函数，用于去除重叠的检测框
def NMS(detectResult):
    predBoxs = []  # 
    # 按照置信度降序排序所有检测框
    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    # 遍历每个检测框
    for i in range(len(sort_detectboxs)):
        # 获取当前检测框的坐标和类别ID
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        # 如果当前检测框未被标记为删除（classId != -1）
        if sort_detectboxs[i].classId != -1:
            # 将当前检测框添加到保留列表
            predBoxs.append(sort_detectboxs[i])
            # 将当前检测框与剩余所有检测框进行比较
            for j in range(i + 1, len(sort_detectboxs), 1):
                # 如果两个检测框属于同一类别
                if classId == sort_detectboxs[j].classId:
                    # 获取待比较检测框的坐标
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    # 计算两个检测框的IOU
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    # 如果IOU大于阈值，则将待比较的检测框标记为删除
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs  # 返回保留的检测框列表


# Sigmoid激活函数，将输入映射到(0,1)区间
def sigmoid(x):
    # 计算sigmoid: f(x) = 1 / (1 + e^(-x))
    # 用于将网络输出转换为概率值
    return 1 / (1 + exp(-x))


# 后处理函数，处理模型的原始输出，转换为检测框结果
def postprocess(out, img_h, img_w):
    # print(meshgrid)
    print('postprocess ... ')
    detectResult = []  # 存储所有检测到的目标
    output = []  # 存储处理后的模型输出
    # 将模型输出展平为一维数组
    for i in range(len(out)):
        # (1, 1, 4, 6400)     x,y,w,h * 80*80
        # (1, 80, 80, 80)      80个类别  80*80
        # (1, 1, 4, 1600)
        # (1, 80, 40, 40)
        # (1, 1, 4, 400)
        # (1, 80, 20, 20)
        output.append(out[i].reshape((-1)))
    # 计算输入图像相对于模型输入尺寸的缩放比例
    scale_h = img_h / input_imgH  # 高度缩放比例
    scale_w = img_w / input_imgW  # 宽度缩放比例
    gridIndex = -2  # 网格索引初始值，用于遍历特征图的每个位置
    # 遍历每个检测头（总共3个检测头，分别处理不同尺度的特征图）
    for index in range(headNum):
        # 获取当前检测头的回归分支和分类分支输出
        reg = output[index * 2 + 0]  # 回归分支输出，包含边界框的坐标信息
        cls = output[index * 2 + 1]  # 分类分支输出，包含每个类别的置信度
        # 打印回归分支和分类分支的数值统计信息，用于调试
        print('reg average, min, max:', np.average(reg), np.min(reg), np.max(reg))
        print('cls average, min, max:', np.average(cls), np.min(cls), np.max(cls))
        
        # 遍历特征图的每个网格点
        for h in range(mapSize[index][0]):  # 遍历特征图的高度
            for w in range(mapSize[index][1]):  # 遍历特征图的宽度
                gridIndex += 2  # 更新网格索引，每个网格点对应两个坐标值(x,y)
                # 遍历所有目标类别
                for cl in range(class_num):
                    # 计算当前网格点对当前类别的置信度
                    # 使用sigmoid函数将输出转换为0-1之间的概率值
                    cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] \
                                          + h * mapSize[index][1] \
                                          + w])
                      # 当前类别的起始位置。
                      # 当前行的起始位置。h 是当前网格点的行索引，mapSize[index][1] 是特征图的宽度。
                      # 前列的具体位置。w 是当前网格点的列索引。

                    # 如果置信度超过阈值，则认为检测到了目标
                    if cls_val > objectThresh:
                        # 计算边界框的坐标（解码过程）
                        # x1,y1为左上角坐标，x2,y2为右下角坐标
                        # meshgrid中存储了特征图上的网格点坐标
                        # reg中存储了相对于网格点的偏移量
                        # strides[index]是当前检测头的步长，用于将特征图坐标映射回原图
                        x1 = (meshgrid[gridIndex + 0] \
                            - reg[0 * mapSize[index][0] * mapSize[index][1] \
                            + h * mapSize[index][1] + w]) 
                                        * strides[index]
# meshgrid: 这是一个列表，存储了特征图上每个网格点的中心坐标。
# gridIndex: 是一个索引变量，用于遍历 meshgrid 中的坐标值。
# gridIndex + 0: 表示当前网格点的 x 坐标（gridIndex + 1 则表示 y 坐标）。
# meshgrid[gridIndex + 0] 表示当前网格点在特征图上的 x 坐标。

# mapSize[index][0] * mapSize[index][1]: 当前检测头的特征图总网格点数。
# h * mapSize[index][1] + w: 当前网格点在 reg 数组中的索引位置。
# 0 * ...: 表示这是 x 方向的偏移量（1 * ... 则表示 y 方向的偏移量，2 * ... 和 3 * ... 分别表示宽度和高度方向的偏移量）。
# 因此，reg[...] 表示当前网格点在 x 方向的偏移量。

# strides: 是一个列表，表示每个检测头的步长（stride）。步长用于将特征图上的坐标映射回原图的尺度。
# strides[index]: 当前检测头的步长。通过乘以 strides[index]，可以将特征图上的坐标转换为原图上的坐标。

                        y1 = (meshgrid[gridIndex + 1] \
                            - reg[1 * mapSize[index][0] * mapSize[index][1] \
                            + h * mapSize[index][1] + w]) * strides[index]
                        x2 = (meshgrid[gridIndex + 0] + reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        y2 = (meshgrid[gridIndex + 1] + reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        # 将边界框坐标缩放到原图尺寸
                        xmin = x1 * scale_w
                        ymin = y1 * scale_h
                        xmax = x2 * scale_w
                        ymax = y2 * scale_h
                        # 确保边界框坐标不超出图像范围
                        xmin = xmin if xmin > 0 else 0
                        ymin = ymin if ymin > 0 else 0
                        xmax = xmax if xmax < img_w else img_w
                        ymax = ymax if ymax < img_h else img_h

                        # 创建检测框对象并添加到结果列表
                        box = DetectBox(cl, cls_val, xmin, ymin, xmax, ymax, index)
                        detectResult.append(box)
    # NMS
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)
    return predBox


# RKNN模型导出和推理函数
def export_rknn_inference(img):
    # 创建RKNN对象，用于模型转换和推理
    rknn = RKNN(verbose=False)  # verbose=False表示不输出详细日志

    # 配置预处理参数
    print('--> Config model')
    rknn.config(
        mean_values=[[0, 0, 0]],  # 图像归一化的均值
        std_values=[[255, 255, 255]],  # 图像归一化的标准差
        quantized_algorithm='normal',  # 量化算法选择
        quantized_method='channel',  # 量化方法选择（按通道量化）
        optimization_level = 2,  # 优化级别设置
        target_platform='rk3588'  # 目标硬件平台
    )
    print('done')

    # 加载ONNX模型
    print('--> Loading model')
    # 指定模型输出节点名称，包括回归分支(reg)和分类分支(cls)
    ret = rknn.load_onnx(model=ONNX_MODEL, 
                         outputs=["reg1","cls1","reg2","cls2","reg3","cls3"])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 构建RKNN模型
    print('--> Building model')
    # do_quantization：是否进行量化
    # dataset：量化校准数据集
    # rknn_batch_size：推理时的批处理大小
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 导出RKNN模型文件
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 初始化运行时环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime()  # 初始化当前平台的运行环境
    # ret = rknn.init_runtime(target='rk3566')  # 也可以指定目标平台
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 执行模型推理
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])  # 输入图像进行推理
    rknn.release()  # 释放RKNN对象
    print('done')

    return outputs  # 返回模型推理结果


# 主函数入口
if __name__ == '__main__':
    print('This is main ...')
    # 生成特征图网格坐标
    GenerateMeshgrid()
    # 读取测试图片
    img_path = './street.jpg'
    orig_img = cv2.imread(img_path)  # 读取原始图像
    img_h, img_w = orig_img.shape[:2]  # 获取原始图像的高度和宽度
    # 图像预处理
    # 1. 将图像缩放到模型输入大小
    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    # 2. 将BGR格式转换为RGB格式
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
    # 3. 添加batch维度
    img = np.expand_dims(origimg, 0)
    # 执行RKNN模型推理
    outputs = export_rknn_inference(img)  # len 6
    # 整理模型输出
    out = outputs[:]
    # 后处理得到检测框结果
    predbox = postprocess(out, img_h, img_w)
    # 打印检测到的目标数量
    print(len(predbox))
    # 在原图上绘制检测结果
    for i in range(len(predbox)):
        # 获取检测框的坐标和信息
        xmin = int(predbox[i].xmin)  # 左上角x坐标
        ymin = int(predbox[i].ymin)  # 左上角y坐标
        xmax = int(predbox[i].xmax)  # 右下角x坐标
        ymax = int(predbox[i].ymax)  # 右下角y坐标
        classId = predbox[i].classId  # 类别ID
        score = predbox[i].score      # 置信度分数
        head = predbox[i].head        # 检测头索引
        # 绘制矩形框，颜色为绿色(BGR格式)
        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 设置文本位置（左上角）
        ptext = (xmin, ymin)
        # 生成显示的文本（类别:检测头:置信度）
        title = CLASSES[classId] + ":%d:%.2f" % (head, score)
        # 绘制文本，颜色为红色
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # 保存结果图像
    cv2.imwrite('./test_rknn_result.jpg', orig_img)
    # 如果需要显示结果，可以取消下面两行的注释
    # cv2.imshow("test", origimg)
    # cv2.waitKey(0)

