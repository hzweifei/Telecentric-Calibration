import cv2
from corner_detector import find_corners,PatternInfo
import numpy as np


pattern_type = PatternInfo(1, (9,9), 0.3, (2.5,2.5),0.15)
image=cv2.imread("1.bmp",cv2.IMREAD_GRAYSCALE)
points=find_corners(image,pattern_type)
points=points[1].reshape(-1,2)
for i in range(len(points)-1):
    p1=points[i]
    p2=points[i+1]
    d=np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    print(d*2.5/1.000985-300)

