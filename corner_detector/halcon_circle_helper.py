import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import numpy as np
from .Pattern_Info import PatternInfo
from .subpixel_edges import subpixel_edges

"""
检测halcon圆形标定板
"""

# 定位halcon标定板的外轮廓黑框
def get_contours(img):
    """
    定位halcon标定板的外轮廓黑框
    :param img: 灰度图像
    :return: 外框的四个点，里框的五个点，以及内部的排序好的圆轮廓
    """
    # 自适应局部二值化
    threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 125, 2)
    # 中值滤波
    median_blurred_image = cv2.medianBlur(threshold, 7)
    # 轮廓提取
    contours, hierarchy = cv2.findContours(median_blurred_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i, contour in enumerate(contours):
        # 计算轮廓内区域的面积
        m = median_blurred_image.shape[0] / 5
        area = cv2.contourArea(contour)
        if area < m:
            continue
        else:
            # 近似多边形
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                # 获取当前轮廓的子轮廓索引
                child_index = hierarchy[0][i][2]
                if child_index != -1 and hierarchy[0][child_index][0] == -1:
                    child_contour = contours[child_index]
                    epsilon = 0.003 * cv2.arcLength(child_contour, True)
                    approx1 = cv2.approxPolyDP(child_contour, epsilon, True)
                    if (len(approx1) == 5):
                        # 提取圆轮廓
                        n, circle_contours = get_all_child_contours(child_index, contours, hierarchy)
                        ret = n
                        # if ret!=w*h
                        return ret, approx.reshape(-1, 2), approx1.reshape(-1, 2), circle_contours
    print("外框定位失败")
    return
# 指定一个reference_point点，数组中的点以该点开始顺时针排序
def clockwise_sort(points, reference_point):
    """
    指定一个reference_point点，数组中的点以该点开始顺时针排序
    :param points:nx2的数组
    :param reference_point:指定点
    :return:排序后的数组
    """
    # 将 points 转换为 numpy 数组
    points = np.array(points)
    # 将 start_point 转换为 numpy 数组
    start_point = np.array(reference_point)
    # 计算正方形中心点
    center = np.mean(points, axis=0)
    # 计算每个点相对于中心点的极角
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    # 按极角排序，顺时针顺序
    sorted_indices = np.argsort(angles)
    # 找到起始点在排序后的索引位置
    for i, point in enumerate(points[sorted_indices]):
        if (point == start_point).all():
            start_index = i
            break
    # 将起始点放在排序后列表的开头
    sorted_points = np.roll(points[sorted_indices], -start_index, axis=0)
    # 返回排序后的点
    return sorted_points
# 将控制点转换到正平面进行排序，从左到右，从上到下。
def order_corners(point4, points, shape_of_corners):
    """
    将控制点转换到正平面进行排序，从左到右，从上到下。
    :param point4: 四个角点
    :param points: 找到的所有控制点
    :param shape_of_corners: 控制点形状
    :return: 排好序后的控制点
    """
    # print(point4)
    w, h = shape_of_corners
    # 在正平面对所有控制点进行排序
    # 正平面的四个点
    point4_frontal = np.array([[0, 0], [100 * w, 0], [100 * w, 100 * h], [0, 100 * h]])
    # 计算变换矩阵
    H, _ = cv2.findHomography(point4, point4_frontal)
    # 将控制点投影到正平面
    points = points.reshape(-1, 1, 2)
    points_frontal = cv2.perspectiveTransform(points, H)
    # 先按y排序，再按x排序
    points_frontal = np.hstack((points_frontal.reshape(-1, 2), points.reshape(-1, 2)))
    sorted_points = points_frontal[np.argsort(points_frontal[:, 1])]
    for i in range(0, len(sorted_points), w):
        s = sorted_points[i:i + w, :]
        s = s[np.argsort(s[:, 0])]
        sorted_points[i:i + w, :] = s
    points = sorted_points[:, 2:4]
    return points
# 获取一个轮廓的所有子轮廓
def get_all_child_contours(contour_index, contours, hierarchy):
    """
    获取一个轮廓的所有子轮廓
    :param contour_index: 轮廓的索引
    :param contours: 找到的轮廓
    :param hierarchy: 关系树
    :return: 所有子轮廓以及总数
    """
    child_contours = []
    # 获取子轮廓数量
    child_count = 0
    child_index = hierarchy[0][contour_index][2]  # 获取第一个子轮廓的索引
    while child_index != -1:
        area = cv2.contourArea(contours[child_index])
        if area > 200:
            child_contours.append(contours[child_index])
            child_count += 1
        child_index = hierarchy[0][child_index][0]  # 获取下一个子轮廓的索引

    return child_count, child_contours

def detect_halcon_corners(gray_img,pattern_info:PatternInfo,threshold=50,show=False):
    """
    Args:
        gray_img: 灰度图像
        pattern_info: 标定板信息
        threshold: 幅度阈值
    Returns: points(nx2)
    """
    # 定位halcon标定板的外轮廓黑框
    ret, point_4, point_5, circle_contours=get_contours(gray_img)
    # 轮廓面积按大小排序，取前w*h个
    w,h=pattern_info.shape
    # 计算每个轮廓的面积
    contour_areas = [(i, cv2.contourArea(circle_contours[i])) for i in range(len(circle_contours))]
    # 按照面积排序
    contour_areas.sort(key=lambda x: x[1], reverse=True)
    # 选择前 w*h 个最大轮廓
    circle_contours = [circle_contours[i[0]] for i in contour_areas[:w * h]]
    # 大致圆中心点
    init_points=[]
    for i in circle_contours:
        # 拟合椭圆
        ellipse = cv2.fitEllipse(i)
        point = ellipse[0]
        init_points.append(point)
    # 控制点排序
    for i, p4 in enumerate(point_4):
        distance = []
        for j, p5 in enumerate(point_5):
            d = np.linalg.norm(p4 - p5)
            distance.append([d, i, j])
        distance = np.array(distance)
        # print(distance)
        sorted_d = distance[np.argsort(distance[:, 0])]
        if (sorted_d[1, 0] / sorted_d[0, 0] < 3):
            p4_1 = p4
            p5_1 = point_5[int(sorted_d[0, 2])]
            p5_2 = point_5[int(sorted_d[1, 2])]
    point_4 = clockwise_sort(point_4, p4_1)
    # 创建mask
    rect = np.zeros(gray_img.shape, dtype=np.uint8)
    cv2.fillPoly(rect, [point_5], 255)  # 填充区域为白色（255）
    # 定义一个3x3的矩形结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 使用腐蚀操作
    eroded_img = cv2.erode(rect, kernel, iterations=2)
    mask=eroded_img>50
    # 亚像素轮廓提取
    e=subpixel_edges(gray_img,threshold,1,2,mask=mask)
    image_draw=cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB)
    for i in e.sub_position:
        image_draw[int(i[1]+0.5),int(i[0]+0.5)]=(255,0,0)
    if ret:
        for i, point in enumerate(point_4):
            cv2.putText(image_draw, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # 亚像素轮廓点分类
    # 创建并拟合 K-means 模型
    kmeans = KMeans(n_clusters=w*h,init=np.array(init_points),n_init="auto")
    kmeans.fit(e.sub_position)
    # 获取每个点的簇标签
    labels = kmeans.labels_
    # 提取每个簇对应的坐标点
    clusters = {}
    center_points=[]
    for i in range(w*h):  # 假设 K=3
        clusters[i] = e.sub_position[labels == i]  # 使用布尔索引获取每个簇的点
        # 使用 OpenCV 的 fitEllipse 函数拟合椭圆
        ellipse = cv2.fitEllipse(clusters[i])
        center = ellipse[0]
        center_points.append(center)
    center_points = order_corners(point_4, np.array(center_points), (w, h))
    if show:
        for i, p in enumerate(center_points):
            cv2.circle(image_draw, (int(p[0]+0.5), int(p[1]+0.5)), 6, (0, 0, 255), 2)
            cv2.putText(image_draw, str(i), (int(p[0]+0.5), int(p[1]+0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        for i in center_points:
            image_draw[int(i[1] + 0.5), int(i[0] + 0.5)] = (255,255, 0)
        plt.imshow(image_draw)
        plt.show()
    return ret,np.array(center_points).astype(np.float32)

