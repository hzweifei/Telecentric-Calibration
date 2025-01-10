from Calibrator.calibrator import Calibrator
from corner_detector import PatternInfo
import time


if __name__=="__main__":
    # ----------HALCON------------------
    img_dir = "image/halcon_circles"
    pattern_type = PatternInfo(1, (9,9), 0.3, (2.5,2.5),0.15)
    print("远心相机开始标定！")
    s=time.time()
    camera = Calibrator(img_dir,pattern_type,1,visualization=False)
    camera.calibrate_camera()
    e=time.time()
    print(f"耗费时间：{e-s}s")
    # -----------CHESSBOARD-------------
    # img_dir = "image/chessboard"
    # pattern_type = PatternInfo(0, (19,17), 0.1, (2.5,2.5),0.15)
    # print("远心相机开始标定！")
    # s=time.time()
    # camera = Calibrator(img_dir,pattern_type,1,visualization=False)
    # camera.calibrate_camera()
    # e=time.time()
    # print(f"耗费时间：{e-s}s")

