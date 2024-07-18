import numpy as np
import cv2
from Pattern_type import PatternType, PatternInfo


def find_corners(image, pattern_info: PatternInfo):
    """
    高精度角度检测
    :param image: 标定图片
    :param pattern_info: 标定板样式
    :return: 角点坐标
    """
    if image.ndim != 2:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 图像不是灰度图
    else:
        gray_img = image
    w, h = pattern_info.shape
    # 检测标定图案信息，根据不同的图案选择不同的控制点检查方法
    if pattern_info.type == PatternType.CHESSBOARD:  # 棋盘格角点检测
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if ret:
            cp_img2 = cv2.cornerSubPix(gray_img, cp_img, (11, 11), (-1, -1), criteria)
            # print(cp_img2.shape)
            # view the corners
            # cv2.drawChessboardCorners(gray_img, (w, h), cp_img2, ret)
            # # 调整图像大小，以适应窗口
            # resized_image = cv2.resize(gray_img, (800, 600))
            # cv2.imshow('FoundCorners', resized_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(cp_img2)
            return cp_img2
        else:
            print("棋盘格角点检测失败")
            return
    if pattern_info.type == PatternType.CIRCLE:  # 圆形圆心检测
        return
    if pattern_info.type == PatternType.RING:  # 圆环检测
        return
