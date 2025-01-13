import cv2
from corner_detector import find_corners,PatternInfo
from Calibrator.calibrator_helper import undistort_points
import numpy as np
from matplotlib import pyplot as plt

coff_dis=[6.371515346120746e-06, -2.5044822129886805e-05, 1.318785672605667e-06,
          0.0001354315719188121, -3.0049662460793236e-05]
mat_intri = np.array([[400.3567797584294, -0.10661893098632821, 1299.5076829378097],
              [0.0, 400.4317198526271, 1079.5206188772452],
              [0.0, 0.0, 1.0]])
m=1.0009856245138207

pattern_type = PatternInfo(0, (19,17), 0.1, (2.5,2.5),0.15)
image=cv2.imread("1.bmp",cv2.IMREAD_GRAYSCALE)
points=find_corners(image,pattern_type)
points=points[1].reshape(-1,2)
# points=undistort_points(points,mat_intri,coff_dis)
cc=[]
for i in range(18):
    p1=points[i]
    p2=points[i+1]
    d=np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    # print(d*2.5/m-100)
    cc.append(abs(d*2.5/m-100))
print("平均误差为：",np.mean(cc))
cc=[]
for i in range(19,37):
    p1=points[i]
    p2=points[i+1]
    d=np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    # print(d*2.5/m-100)
    cc.append(abs(d*2.5/m-100))
print("平均误差为：",np.mean(cc))
cc=[]
for i in range(38,56):
    p1=points[i]
    p2=points[i+1]
    d=np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    # print(d*2.5/m-100)
    cc.append(abs(d*2.5/m-100))
print("平均误差为：",np.mean(cc))

print("未矫正畸变前")
p1=points[18]
p2=points[304]
cv2.circle(image,(int(p1[0]),int(p1[1])),5,(255,0,0))
cv2.circle(image,(int(p2[0]),int(p2[1])),5,(255,0,0))
plt.imshow(image)
plt.show()
d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
print(d*2.5/m - np.sqrt((18*100)**2+(16*100)**2))
print("矫正畸变后")
p=undistort_points(np.array([p1,p2]),mat_intri,coff_dis)
p1=p[0]
p2=p[1]
d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
print(d*2.5/m - np.sqrt((18*100)**2+(16*100)**2))




