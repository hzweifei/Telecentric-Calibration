from modules import *
from corner_detector import high_accuracy_corner_detector
from corner_detector import Pattern_type
from Calibrator import calibrator_helper


class Calibrator:
    def __init__(self, img_dir, pattern_type: Pattern_type.PatternInfo, m, visualization=False):
        """
        :param img_dir: 标定图片文件夹
        :param pattern_type: 标定板样式
        :param m: 远心镜头放大倍数
        :param visualization: 是否可视化
        """
        self.pattern_type = pattern_type
        self.m = m
        self.visualization = visualization
        self.mat_intri = None  # intrinsic matrix
        self.coff_dis = None  # cofficients of distortion
        self.v_rot = None  # 旋转向量
        self.v_trans = None  # 位移向量
        self.json_data = None
        # 控制点的像素坐标
        self.points_pixel = None
        # 生成标定板的世界坐标
        w, h = pattern_type.shape
        # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ...., (10,7,0)
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # cp_world: corner point in world space, save the coordinate of corner points in world space
        self.cp_world = cp_int * pattern_type.distance
        # 标定板图片
        ret, self.img_paths = calibrator_helper.get_sorted_image_paths(img_dir)
        if ret == 0:
            raise ValueError("图片呢，你这混蛋！")

    # 标定相机
    def calibrate_camera(self):
        points_world = []  # the points in world space
        points_pixel = []  # the points in pixel space (relevant to points_world)
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,cp_img2 = high_accuracy_corner_detector.find_corners(gray_img, self.pattern_type)
            points_world.append(self.cp_world)
            points_pixel.append(cp_img2)
            # print(points_pixel)
        """
        远心成像模型可以近似为光心在无穷远处的小孔成像模型，可以根据外参不变性原理
        解决远心成像模型旋转矩阵歧义的问题
        """

        # 针孔相机标定得到外参
        ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1],
                                                                       None, None)
        # print(mat_intri)
        if ret:
            print("获取外参成功，针孔模型重投影误差为：", ret)
            # print(np.array(v_trans).reshape(-1,3),np.array(v_trans))
            self.v_rot = np.array(v_rot).reshape(-1, 3)
            # 远心成像缺少t_z
            self.v_trans = np.array(v_rot).reshape(-1, 3)[:, :2]
        else:
            print("针孔相机标定出错")
            return
        # 远心成像模型内参初始化
        dx, dy = self.pattern_type.pixel_size
        dx = dx / 1000
        dy = dy / 1000
        u0, v0 = mat_intri[0, 2], mat_intri[1, 2]
        self.mat_intri = np.array([[self.m / dx, 0, u0],
                                   [0, self.m / dy, v0],
                                   [0, 0, 1]])
        # 优化不带畸变的远心成像模型参数
        ret, mat_intri, v_rot, v_trans = calibrator_helper.refine_params_without_distortion(points_world, points_pixel,
                                                                                            self.mat_intri, self.v_rot,
                                                                                            self.v_trans)
        if ret:
            print("优化初始参数成功(无畸变），重投影误差为：", ret)
            # print("相机内参：",mat_intri)
        else:
            print("优化初始参数出错")
        # 优化带畸变的远心成像模型参数
        # 初始化畸变系数(远心镜头暂时只考虑径向畸变）
        coff_dis = [0, 0]
        ret, mat_intri, coff_dis, v_rot, v_trans = calibrator_helper.refine_params_with_distortion(points_world,
                                                                                                   points_pixel,
                                                                                                   mat_intri, coff_dis,
                                                                                                   v_rot,
                                                                                                   v_trans)
        if ret:
            print("优化初始参数成功(带畸变），重投影误差为：", ret)
            # print("相机内参：",mat_intri)
        else:
            print("优化初始参数出错")
        self.m = (mat_intri[0, 0] + mat_intri[1, 1]) / 2 * dx
        self.points_pixel = points_pixel
        self.mat_intri = mat_intri
        self.coff_dis = coff_dis
        self.v_rot = v_rot
        self.v_trans = v_trans
        # 要写入的数据
        data = {
            'Magnification': self.m,
            'mat_intri': self.mat_intri.tolist(),
            'coff_dis': self.coff_dis,
            'v_rot': self.v_rot.tolist(),
            'v_trans': self.v_trans.tolist()
        }
        json_file_path = 'calibrate_result.json'
        # 写入 JSON 文件
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
            print("标定成功，结果保存在calibrate_result.json中")
