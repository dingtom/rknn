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
# 导入exp函数，用于计算指数
from math import exp

# 定义ONNX模型的路径
ONNX_MODEL = './coco_pose_s_relu.onnx'
# 定义导出的RKNN模型的路径
RKNN_MODEL = './coco_pose_s_relu_int8.rknn'
# 备注：可以使用 './yolov8-pose-int.rknn' 作为RKNN模型的路径
# RKNN_MODEL = './yolov8-pose-int.rknn'
# 定义数据集文件的路径，用于量化
DATASET = './datasets.txt'

# 定义是否开启量化
QUANTIZE_ON = True

# 定义头部、躯干、四肢的颜色
color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
# 定义2D姿态关键点的连接关系
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# 定义类别列表
CLASSES = ['person']

# 定义网格点列表
meshgrid = []

# 定义类别数量
class_num = len(CLASSES)
# 定义头部数量
headNum = 3
# 定义关键点数量
keypoint_num = 17

# 定义步长列表
strides = [8, 16, 32]
# 定义特征图尺寸列表
mapSize = [[80, 80], [40, 40], [20, 20]]
# 定义非极大值抑制阈值
nmsThresh = 0.55
# 定义目标检测阈值
objectThresh = 0.5

# 定义输入图像的高度
input_imgH = 640
# 定义输入图像的宽度
input_imgW = 640


# 定义检测框类
class DetectBox:
    # 初始化方法，接收类别ID、得分、边界框坐标和姿态信息
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, pose):
        # 类别ID
        self.classId = classId
        # 得分
        self.score = score
        # 边界框左上角x坐标
        self.xmin = xmin
        # 边界框左上角y坐标
        self.ymin = ymin
        # 边界框右下角x坐标
        self.xmax = xmax
        # 边界框右下角y坐标
        self.ymax = ymax
        # 姿态信息
        self.pose = pose


# 生成网格点
def GenerateMeshgrid():
    # 遍历每个头部
    for index in range(headNum):
        # 遍历特征图的每个高度
        for i in range(mapSize[index][0]):
            # 遍历特征图的每个宽度
            for j in range(mapSize[index][1]):
                # 添加网格点的x坐标
                meshgrid.append(j + 0.5)
                # 添加网格点的y坐标
                meshgrid.append(i + 0.5)


# 计算两个边界框的交并比（IOU）
def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    # 计算交集的左上角x坐标
    xmin = max(xmin1, xmin2)
    # 计算交集的左上角y坐标
    ymin = max(ymin1, ymin2)
    # 计算交集的右下角x坐标
    xmax = min(xmax1, xmax2)
    # 计算交集的右下角y坐标
    ymax = min(ymax1, ymax2)

    # 计算交集的宽度
    innerWidth = xmax - xmin
    # 计算交集的高度
    innerHeight = ymax - ymin

    # 如果宽度小于等于0，则设置为0
    innerWidth = innerWidth if innerWidth > 0 else 0
    # 如果高度小于等于0，则设置为0
    innerHeight = innerHeight if innerHeight > 0 else 0

    # 计算交集的面积
    innerArea = innerWidth * innerHeight

    # 计算第一个边界框的面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    # 计算第二个边界框的面积
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算总面积
    total = area1 + area2 - innerArea

    # 返回交并比
    return innerArea / total


# 非极大值抑制（NMS）
def NMS(detectResult):
    # 定义预测框列表
    predBoxs = []

    # 按得分降序排序检测结果
    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    # 遍历排序后的检测结果
    for i in range(len(sort_detectboxs)):
        # 获取当前检测框的边界框坐标
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        # 获取当前检测框的类别ID
        classId = sort_detectboxs[i].classId

        # 如果当前检测框的类别ID不为-1
        if sort_detectboxs[i].classId != -1:
            # 将当前检测框添加到预测框列表
            predBoxs.append(sort_detectboxs[i])
            # 遍历剩余的检测结果
            for j in range(i + 1, len(sort_detectboxs), 1):
                # 如果类别ID相同
                if classId == sort_detectboxs[j].classId:
                    # 获取下一个检测框的边界框坐标
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    # 计算两个检测框的交并比
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    # 如果交并比大于阈值
                    if iou > nmsThresh:
                        # 将下一个检测框的类别ID设置为-1
                        sort_detectboxs[j].classId = -1
    # 返回预测框列表
    return predBoxs


# 定义sigmoid函数
def sigmoid(x):
    # 返回sigmoid函数的值
    return 1 / (1 + exp(-x))


# 后处理函数
def postprocess(out, img_h, img_w):
    # 打印后处理开始信息
    print('postprocess ... ')

    # 定义检测结果列表
    detectResult = []

    # 定义输出列表
    output = []
    # 遍历输出
    for i in range(len(out)):
        # 将输出展平并添加到输出列表
        output.append(out[i].reshape((-1)))

    # 计算高度缩放比例
    scale_h = img_h / input_imgH
    # 计算宽度缩放比例
    scale_w = img_w / input_imgW

    # 定义网格索引
    gridIndex = -2
    # 遍历每个头部
    for index in range(headNum):
        # 获取回归框的坐标
        reg = output[index * 2 + 0]
        # 获取类别置信度
        cls = output[index * 2 + 1]
        # 获取关键点的坐标
        pose = output[headNum * 2 + index]
        # 遍历每个头部的每个格子，先遍历高，再遍历宽
        for h in range(mapSize[index][0]):
            for w in range(mapSize[index][1]):
                # 更新网格索引
                gridIndex += 2
                # 遍历每个类别
                for cl in range(class_num):
                    # 获取类别的置信度
                    cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])
                    
                    # 如果置信度大于目标检测阈值
                    if cls_val > objectThresh:
                        # 计算回归框的左上角x坐标
                        x1 = (meshgrid[gridIndex + 0] - reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        # 计算回归框的左上角y坐标
                        y1 = (meshgrid[gridIndex + 1] - reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        # 计算回归框的右下角x坐标
                        x2 = (meshgrid[gridIndex + 0] + reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        # 计算回归框的右下角y坐标
                        y2 = (meshgrid[gridIndex + 1] + reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]

                        # 计算边界框的左上角x坐标
                        xmin = x1 * scale_w
                        # 计算边界框的左上角y坐标
                        ymin = y1 * scale_h
                        # 计算边界框的右下角x坐标
                        xmax = x2 * scale_w
                        # 计算边界框的右下角y坐标
                        ymax = y2 * scale_h

                        # 如果边界框的左上角x坐标小于等于0，则设置为0
                        xmin = xmin if xmin > 0 else 0
                        # 如果边界框的左上角y坐标小于等于0，则设置为0
                        ymin = ymin if ymin > 0 else 0
                        # 如果边界框的右下角x坐标大于图像宽度，则设置为图像宽度
                        xmax = xmax if xmax < img_w else img_w
                        # 如果边界框的右下角y坐标大于图像高度，则设置为图像高度
                        ymax = ymax if ymax < img_h else img_h

                        # 定义姿态结果列表
                        poseResult = []
                        # 遍历每个关键点
                        for kc in range(keypoint_num):
                            # 获取关键点的x坐标
                            px = pose[(kc * 3 + 0) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]  
                            # 获取关键点的y坐标
                            py = pose[(kc * 3 + 1) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]
                            # 获取关键点的置信度
                            vs = sigmoid(pose[(kc * 3 + 2) * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w])

                            # 计算关键点的x坐标
                            x = (px * 2.0 + (meshgrid[gridIndex + 0] - 0.5)) * strides[index] * scale_w
                            # 计算关键点的y坐标
                            y = (py * 2.0 + (meshgrid[gridIndex + 1] - 0.5)) * strides[index] * scale_h

                            # 将关键点的置信度、x坐标和y坐标添加到姿态结果列表
                            poseResult.append(vs)
                            poseResult.append(x)
                            poseResult.append(y)
                        # 打印姿态结果
                        # print(poseResult)
                        # 创建检测框对象
                        box = DetectBox(cl, cls_val, xmin, ymin, xmax, ymax, poseResult)
                        # 将检测框对象添加到检测结果列表
                        detectResult.append(box)

    # 打印检测结果数量
    print('detectResult:', len(detectResult))
    # 执行非极大值抑制
    predBox = NMS(detectResult)

    # 返回预测框
    return predBox


# 导出RKNN模型并进行推理
def export_rknn_inference(img):
    # 创建RKNN对象
    rknn = RKNN(verbose=True)

    # 配置模型
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    # 加载ONNX模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=["reg1", "cls1", "reg2", "cls2", "reg3", "cls3", "ps1", "ps2", "ps3"])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 构建模型
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
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
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # 运行模型
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    # 释放资源
    rknn.release()
    print('done')

    # 返回输出
    return outputs


# 主函数
if __name__ == '__main__':
    # 打印主函数开始信息
    print('This is main ...')
    # 生成网格点
    GenerateMeshgrid()
    # 定义图像路径
    img_path = './yoga.jpg'
    # 读取图像
    orig_img = cv2.imread(img_path)
    # 获取图像的高度和宽度
    img_h, img_w = orig_img.shape[:2]


    # 调整图像大小
    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    # 转换颜色空间
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)

    # 扩展图像维度
    img = np.expand_dims(origimg, 0)

    # 导出RKNN模型并进行推理
    outputs = export_rknn_inference(img)

    # 定义输出列表
    out = []
    # 遍历输出
    for i in range(len(outputs)):
        # 将输出添加到输出列表
        out.append(outputs[i])
    # 后处理
    predbox = postprocess(out, img_h, img_w)

    # 打印检测到的对象数量
    print('obj num is :', len(predbox))

    # 遍历预测框
    for i in range(len(predbox)):
        # 获取边界框的坐标
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        # 获取类别ID
        classId = predbox[i].classId
        # 获取得分
        score = predbox[i].score

        # 绘制边界框
        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (128, 128, 0), 2)
        # 定义文本位置
        ptext = (xmin, ymin)
        # 定义标题
        title = CLASSES[classId] + "%.2f" % score
        # 绘制文本
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 获取姿态信息
        pose = predbox[i].pose
        # 绘制2点之间的连线
        for i, sk in enumerate(skeleton):
            # 如果置信度大于0.5
            if pose[(sk[0] - 1) * 3] > 0.5:
                # 获取连接的第一个点的坐标
                pos1 = (int(pose[(sk[0] - 1) * 3 + 1]), int(pose[(sk[0] - 1) * 3 + 2]))
                # 获取连接的第二个点的坐标
                pos2 = (int(pose[(sk[1] - 1) * 3 + 1]), int(pose[(sk[1] - 1) * 3 + 2]))
                # 根据关键点ID选择颜色
                if (sk[0] - 1) < 5:
                    color = color_list[0]
                elif 5 <= sk[0] - 1 < 12:
                    color = color_list[1]
                else:
                    color = color_list[2]
                # 绘制线
                cv2.line(orig_img, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)
        # 绘制关键点
        for i in range(0, keypoint_num):
            # 如果置信度大于0.5
            if pose[i * 3] > 0.5:
                # 根据关键点ID选择颜色
                if i < 5:
                    color = color_list[0]
                elif 5 <= i < 12:
                    color = color_list[1]
                else:
                    color = color_list[2]
                # 绘制圆点
                cv2.circle(orig_img, (int(pose[i * 3 + 1]), int(pose[i * 3 + 2])), 2, color, 5)
                # 绘制ID
                cv2.putText(orig_img, str(i), (int(pose[i * 3 + 1]) + 5, int(pose[i * 3 + 2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        

    # 保存结果图像
    cv2.imwrite('./test_rknn_result.jpg', orig_img)