from .Pattern_Info import PatternInfo
from .chessboard_helper import detect_chessboard_corners
from .halcon_circle_helper import detect_halcon_corners
import cv2
import numpy as np


__all__ = ['PatternInfo', 'find_corners','refine_corners']

def find_corners(image, pattern_info: PatternInfo,visualization=False):
    """
    高精度角度检测
    :param image: 标定图片
    :param pattern_info: 标定板样式
    :param visualization: 是否显示找到的控制点
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
        ret,points=detect_halcon_corners(gray_img,pattern_info,show=visualization)
        return ret,points
def refine_corners(image,corners,shape_inner_corner,K,D):
    """
    @brief 优化控制点
    :param image:图像数据
    :param corners:控制点numpy数组
    :param shape_inner_corner:内角的形状，Array of int， (h, w)
    :param pattern_infos:标定图案信息
    :param K:初步标定得到的内参
    :param D:初步标定得到的畸变参数
    :return:优化过后控制点的一维numpy数组
    """
    if image.ndim != 2:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 图像不是灰度图
    else:
        gray_img=image
    w,h=shape_inner_corner
    # 计算最优新相机内参矩阵
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(K, D, image.shape[:2], 1)
    # 1.图像去畸变
    img_und=cv2.undistort(gray_img,K,D,None,new_camera_matrix)
    # 2.控制点去畸变
    corners_und=cv2.undistortPoints(corners.reshape(-1, 1, 2), K, D, P=new_camera_matrix).reshape(-1,2)
    # 3.选取去畸变控制点的四个角点估计单应性矩阵H
    corners_of_corners=np.asarray([corners_und[0],corners_und[w-1],corners_und[w*(h-1)],corners_und[w*h-1]]).reshape(-1,2)
    #print(corners_of_corners)
    size=30
    corners_of_corners_frontal =np.asarray([[50,50],[50+(w-1)*size,50],[50,50+(h-1)*size],[50+(w-1)*size,50+(h-1)*size]])
    #print(corners_of_corners_frontal)
    # 4.应用H变换图像
    H=cv2.findHomography(corners_of_corners,corners_of_corners_frontal)
    H=H[0]
    img_frontal=cv2.warpPerspective(img_und,H,img_und.shape)
    # 显示校正后的图像
    # cv2.imshow("Undistorted Image", img_frontal)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 5.变换后的图像找控制点
    corners_refine_frontal=find_corners(img_frontal,shape_inner_corner,1)
    # 6.将控制点变换为原图像的坐标
    corners_refine_without_distortion=cv2.perspectiveTransform(corners_refine_frontal,np.linalg.inv(H)).reshape(-1,2)
    # print("优化过后的原图像坐标",corners_refine_without_distortion)
    # print("原图像坐标",corners_und)
    # 7.加畸变
    corners_refined_without_distortion_3d=[]
    fx=new_camera_matrix[0,0]
    fy=new_camera_matrix[1,1]
    cx=new_camera_matrix[0,2]
    cy=new_camera_matrix[1,2]
    for i in corners_refine_without_distortion:
        #print(i)
        x = i[0]
        y = i[1]
        corners_refined_without_distortion_3d.append([(x-cx) / fx,(y-cy) / fy,1])
    corners_refined_without_distortion_3d=np.asarray(corners_refined_without_distortion_3d)
    # print(corners_refined_without_distortion_3d)
    corners_refined=cv2.projectPoints(corners_refined_without_distortion_3d,(0, 0, 0), (0, 0, 0), K, D)
    # cv2.projectPoints的返回值有两个array
    corners_refined=corners_refined[0].astype(np.float32)
    #print(corners_refined)
    return corners_refined