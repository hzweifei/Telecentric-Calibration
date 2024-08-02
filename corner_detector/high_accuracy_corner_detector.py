from modules import *
from corner_detector.Pattern_type import *


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
    if pattern_info.type.value == PatternType.CHESSBOARD.value:  # 棋盘格角点检测
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if ret:
            cp_img2 = cv2.cornerSubPix(gray_img, cp_img, (11, 11), (-1, -1), criteria)
            return ret, cp_img2
        else:
            print("棋盘格角点检测失败")
            return
    if pattern_info.type.value == PatternType.CIRCLE.value:  # 圆形圆心检测
        return
    if pattern_info.type.value == PatternType.RING.value:  # 圆环检测
        return
    if pattern_info.type.value == PatternType.HALCON_CIRCLE.value:  # halcon圆检测

        




        return
