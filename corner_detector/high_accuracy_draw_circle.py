import cv2
import matplotlib.pyplot as plt
import numpy as np


# 画一个精细的圆
def draw_circle(image, center, radius, value):
    """
    画一个圆，带锯齿
    :param image: 灰度图
    :param center: 圆心坐标(x,y)，以横轴为 x，纵轴为 y。
    :param radius: 半径，单位像素
    :param value: 圆内的像素值
    :return: img
    """
    image_copy = image.copy()
    for i in range(int(center[0] - radius - 1), int(center[0] + radius) + 1):
        for j in range(int(center[1] - radius - 1), int(center[1] + radius + 1)):
            ret = (i - center[0]) ** 2 + (j - center[1]) ** 2 - radius ** 2
            # 在圆内
            if ret <= 0:
                image_copy[j, i] = value

    return image_copy


def draw_accurate_circle(image, center, radius):
    """
    画一个精细的圆
    :param image: 灰度图
    :param center: 圆心坐标(x,y)，以横轴为 x，纵轴为 y。
    :param radius: 半径，单位像素
    :return: img
    """
    center_x = center[0]
    center_y = center[1]
    img = image.copy()
    img_255 = draw_circle(img, center, radius, 255)
    img_1 = draw_circle(img, center, radius, 1)
    # 找到圆的边界轮廓
    for i in range(int(center_x - radius - 3), int(center_x + radius + 3)):
        for j in range(int(center_y - radius - 3), int(center_y + radius + 3)):
            ret = img_1[j - 1, i - 1] + img_1[j, i - 1] + img_1[j + 1, i - 1] + img_1[j - 1, i] + \
                  img_1[j + 1, i] + img_1[j - 1, i + 1] + \
                  img_1[j, i + 1] + img_1[j + 1, i + 1]
            if 8 > ret > 0:
                # 该点是边界点
                h = 0
                # 生成以当前边界点为中心的像素网格
                # print(len(np.arange(- 0.475, 0.5, 0.05)))
                for m in np.arange(- 0.475, 0.5, 0.05):
                    for n in np.arange(-0.475, 0.5, 0.05):
                        pixel_point = [i + m, j + n]
                        # print(pixel_point)
                        # 计算像素点相对于椭圆中心的距离，判断多少个像素点在椭圆内
                        single_distance = (pixel_point[0] - center_x) ** 2 + (pixel_point[1] - center_y) ** 2
                        if single_distance <= radius ** 2:
                            h = h + 1
                img_255[j, i] = np.uint8((h / 400) * 255)
    return img_255


def draw_circle_pattern(image, shape_inner_corner, radius, distance):
    """
    绘制标定板
    :param image: 灰度图像
    :param shape_inner_corner: 控制点形状，如 9x9。
    :param radius: 半径，以像素为单位。
    :param distance: 控制点之间的距离
    :return:img
    """
    w, h = shape_inner_corner
    start_point = (int(radius) + 50, int(radius) + 50)
    points_list = []
    for i in range(w):
        for j in range(h):
            points_list.append((start_point[0] + distance * i, start_point[1] + distance * j))
    img = image.copy()
    for i in points_list:
        img = draw_accurate_circle(img, i, radius)

    return img

if __name__ == "__main__":
    # 创建一个空白灰度图像
    img = np.zeros((250, 250), dtype=np.uint8)
    # 调用绘制椭圆函数
    img_with_circle = draw_accurate_circle(img, (120.67, 130.35), 100.5)
    cv2.imwrite("circle.jpg", img_with_circle)

    # # 绘制标定图案
    # img_2=draw_circle_pattern(img,(7,7),30,100)
    # plt.imshow(img_2)
    # plt.show()
    # cv2.imwrite("board.jpg",img_2)
