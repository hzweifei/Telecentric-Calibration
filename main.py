from Calibrator.calibrator import Calibrator
from corner_detector import PatternInfo

img_dir = "image/chessboard"
pattern_type = PatternInfo(0, (19, 17), 0.5, (4.8,4.8))
print("远心相机开始标定！")
camera = Calibrator(img_dir,pattern_type,0.4)
camera.calibrate_camera()
print("LOL")

