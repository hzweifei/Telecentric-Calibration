import cv2
from .Pattern_Info import PatternInfo
def detect_chessboard_corners(gray_img,pattern_info:PatternInfo):
    # 控制点形状
    w,h=pattern_info.shape
    """ 检测棋盘格角点 """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if ret:
        cp_img2 = cv2.cornerSubPix(gray_img, cp_img, (11, 11), (-1, -1), criteria)
        return ret, cp_img2
    else:
        print("棋盘格角点检测失败")
        return False, None
