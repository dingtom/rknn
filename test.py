'''
Author: dingtom 35419759+dingtom@users.noreply.github.com
Date: 2025-03-27 23:43:47
LastEditors: dingtom 35419759+dingtom@users.noreply.github.com
LastEditTime: 2025-03-27 23:47:01
FilePath: \convert_rknn\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
strides = [8, 16, 32]  # 三个检测头的步长，用于计算特征图的缩放比例
mapSize = [[80, 80], [40, 40], [20, 20]]  # 三个检测头的特征图大小
headNum = 3
meshgrid = []  
# 遍历每个检测头
for index in range(headNum):
    # 遍历特征图的每个网格点
    for i in range(mapSize[index][0]):  # 遍历高度
        for j in range(mapSize[index][1]):  # 遍历宽度
            # 将网格点的x、y坐标添加到meshgrid列表中
            # 加0.5是为了将坐标对齐到网格中心
            meshgrid.append(j + 0.5)  # x坐标
            meshgrid.append(i + 0.5)  # y坐标
    print(len(meshgrid))