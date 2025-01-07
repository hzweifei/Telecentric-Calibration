from .Pattern_Info import PatternInfo
from .chessboard_helper import detect_chessboard_corners
from .halcon_circle_helper import detect_halcon_corners
import cv2

__all__ = ['PatternInfo', 'find_corners']

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
    # 检测标定图案信息，根据不同的图案选择不同的控制点检查方法
    if pattern_info.pattern_type == 0:  # 棋盘格角点检测
        ret,points=detect_chessboard_corners(gray_img,pattern_info)
        return ret,points
    if pattern_info.pattern_type == 1:  # halcon圆检测
        ret,points=detect_halcon_corners(gray_img,pattern_info)
        return ret,points
