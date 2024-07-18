from calibrator import Calibrator
import cv2
from Pattern_type import *


img_dir = "image/chessboard"
pattern_type = PatternInfo(PatternType.CHESSBOARD, (19, 17), 0.5, (4.8,4.8))
camera = Calibrator(img_dir,pattern_type,0.4)
camera.calibrate_camera()
print("LOL")

