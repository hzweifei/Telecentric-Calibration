import glob
import os
import numpy as np
import cv2
from scipy.optimize import curve_fit


# 得到给定文件夹下所有图片路径，按名称排序。
def get_sorted_image_paths(folder_path):
    """
    得到给定文件夹下所有图片路径，按名称排序。
    :param folder_path: 文件夹路径
    :return: 图片路径列表
    """
    # 获取所有图片文件路径
    image_paths = glob.glob(os.path.join(folder_path, '*'))

    # 过滤出图片文件（根据扩展名）
    image_paths = [path for path in image_paths if
                   path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    if len(image_paths) == 0:
        ret = 0
    else:
        ret = 1
        # 按文件名排序
        image_paths.sort(key=lambda x: os.path.basename(x))
    return ret, image_paths


# 计算坐标数组的齐次坐标
def convert_to_homogeneous_2d_or_3d(points):
    """
    计算nx2或nx3 numpy 坐标数组的齐次坐标
    Args:
        points: nx2或nx3坐标数组
    Returns:
        nx3或nx4齐次坐标数组
    """
    # Check if input is a Nx2 array or Nx3 array
    if points.shape[1] == 2 or points.shape[1] == 3:
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points, ones))
    else:
        raise ValueError("齐次坐标转换失败，输入应该是nx2或nx3坐标数组")
    return homogeneous_points


# 投影点加畸变（远心镜头畸变较小，暂时考虑两个畸变参数）
def distort(k, normalized_proj):
    """
    投影点加畸变
    Args:
        k: 畸变系数
        normalized_proj: 投影坐标点(nx3)：成像平面齐次坐标。

    Returns:
        带畸变的投影点数组
    """

    x, y = normalized_proj[:, 0], normalized_proj[:, 1]

    # Calculate radii
    r = np.sqrt(x ** 2 + y ** 2)

    k0, k1 = k

    # Calculate distortion effects
    D = k0 * r ** 2 + k1 * r ** 4

    # Calculate distorted normalized projection values
    x_prime = x * (1. + D)
    y_prime = y * (1. + D)

    distorted_proj = np.hstack((x_prime[:, np.newaxis], y_prime[:, np.newaxis]))
    distorted_proj = convert_to_homogeneous_2d_or_3d(distorted_proj)
    return distorted_proj


# 优化内参和所有外参（不带畸变）
def refine_params_without_distortion(points_world, points_pixel, mat_intri, v_rot, v_trans):
    """
    优化内参和所有外参
    Args:
        points_world: 控制点世界坐标
        points_pixel: 控制点像素坐标
        mat_intri: 相机内参3x3
        v_rot: 旋转向量nx3
        v_trans: 位移向量nx2

    Returns:
        重投影误差，优化后的内参，所有外参。
    """
    points_pixel = np.array(points_pixel)
    # print(points_pixel,points_pixel.reshape(-1))
    points_world = np.array(points_world)
    # 打包所有参数
    packed_params = []
    # 5个内参
    alpha, beta, gamma, u_c, v_c = mat_intri[0, 0], mat_intri[1, 1], mat_intri[0, 1], mat_intri[0, 2], mat_intri[1, 2]
    packed_params.extend([alpha, beta, gamma, u_c, v_c])
    # 打包所有外参
    for i in range(len(v_rot)):
        rho_x, rho_y, rho_z = v_rot[i]
        t_x, t_y = v_trans[i]
        e = [rho_x, rho_y, rho_z, t_x, t_y]
        packed_params.extend(e)
    # 设置边界约束
    min_bounds = [-np.inf] * len(packed_params)
    max_bounds = [np.inf] * len(packed_params)
    min_bounds[3], max_bounds[3] = u_c - 2, u_c + 2
    min_bounds[4], max_bounds[4] = v_c - 2, v_c + 2
    bounds = (min_bounds, max_bounds)

    # 对参数进行优化
    def project(x_data, *params):
        # 不带畸变的投影
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2] = params[:5]
        v_RT = params[5:]
        y_pre_list = []
        for i in range(len(x_data)):
            world = np.array(x_data[i]).reshape(-1, 3)
            # 齐次坐标
            world[:, 2] = 1
            rt = v_RT[i * 5:(i + 1) * 5]
            # 使用Rodrigues公式将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rt[:3])
            rt_matri = np.eye(3)
            rt_matri[:2, :2] = rotation_matrix[:2, :2]
            rt_matri[:2, 2] = rt[3:5]
            # 投影三维空间点到图像平面上
            y_pre = (K @ rt_matri @ world.T).T
            y_pre = y_pre[:, :2]
            y_pre_list.append(y_pre)
        y_pre_list = np.array(y_pre_list).reshape(-1)
        return y_pre_list

    popt, pcov = curve_fit(project, points_world, points_pixel.reshape(-1), packed_params, bounds=bounds, maxfev=20000)
    # 解包所有参数
    params_refined = popt
    intrinsics = params_refined[:5]
    # 相机内参K
    alpha, beta, gamma, u_c, v_c = intrinsics
    K = np.array([[alpha, gamma, u_c],
                  [0., beta, v_c],
                  [0., 0., 1.]])
    # 所有外参
    rt_v = params_refined[5:]
    m = int(len(rt_v) / 5)
    v_rot = []
    v_trans = []
    for i in range(m):
        v_rot.append(rt_v[i * 5:i * 5 + 3])
        v_trans.append(rt_v[i * 5 + 3:(i + 1) * 5])
    v_rot = np.array(v_rot)
    v_trans = np.array(v_trans)

    # 计算重投影误差
    loss_list = []
    for i in range(len(points_world)):
        world = points_world[i].reshape(-1, 3)
        world[:, 2] = 1
        pixel = points_pixel[i].reshape(-1, 2)
        r_m, _ = cv2.Rodrigues(v_rot[i])
        rt_matri = np.eye(3)
        rt_matri[:2, :2] = r_m[:2, :2]
        rt_matri[:2, 2] = v_trans[i].T
        # 投影三维空间点到图像平面上
        y_pre = (K @ rt_matri @ world.T).T
        loss = np.linalg.norm(y_pre[:, :2] - pixel, axis=1)
        loss_list.append(np.mean(loss))
    ret = np.mean(loss_list)
    return ret, K, v_rot, v_trans


# 优化内参和所有外参（带畸变）
def refine_params_with_distortion(points_world, points_pixel, mat_intri, coff_dis, v_rot, v_trans):
    """
    优化内参，畸变系数，所有外参
    Args:
        points_world: 控制点世界坐标
        points_pixel: 控制点像素坐标
        mat_intri: 相机内参3x3
        coff_dis: 畸变系数
        v_rot: 旋转向量nx3
        v_trans: 位移向量nx2
    """
    points_pixel = np.array(points_pixel)
    # print(points_pixel,points_pixel.reshape(-1))
    points_world = np.array(points_world)
    # 打包所有参数
    packed_params = []
    # 5个内参
    alpha, beta, gamma, u_c, v_c = mat_intri[0, 0], mat_intri[1, 1], mat_intri[0, 1], mat_intri[0, 2], mat_intri[1, 2]
    # 两个外参
    k1, k2 = coff_dis[0], coff_dis[1]
    packed_params.extend([alpha, beta, gamma, u_c, v_c, k1, k2])
    # 打包外参
    for i in range(len(v_rot)):
        rho_x, rho_y, rho_z = v_rot[i]
        t_x, t_y = v_trans[i]
        e = [rho_x, rho_y, rho_z, t_x, t_y]
        packed_params.extend(e)
    # 设置边界约束
    min_bounds = [-np.inf] * len(packed_params)
    max_bounds = [np.inf] * len(packed_params)
    min_bounds[3], max_bounds[3] = u_c - 2, u_c + 2
    min_bounds[4], max_bounds[4] = v_c - 2, v_c + 2
    bounds = (min_bounds, max_bounds)

    def project(x_data, *params):
        K = np.eye(3)
        K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], k1, k2 = params[:7]
        coff_dis = np.array([k1, k2])
        v_RT = params[7:]
        y_pre_list = []
        for i in range(len(x_data)):
            world = np.array(x_data[i]).reshape(-1, 3)
            # 齐次坐标
            world[:, 2] = 1
            rt = v_RT[i * 5:(i + 1) * 5]
            # 使用Rodrigues公式将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rt[:3])
            rt_matri = np.eye(3)
            rt_matri[:2, :2] = rotation_matrix[:2, :2]
            rt_matri[:2, 2] = rt[3:5]
            # 投影三维空间点到成像平面上
            y_pre = (rt_matri @ world.T).T
            y_dis = distort(coff_dis, y_pre)
            y_pre = (K @ y_dis.T).T
            y_pre = y_pre[:, :2]
            y_pre_list.append(y_pre)
        y_pre_list = np.array(y_pre_list).reshape(-1)
        return y_pre_list

    popt, pcov = curve_fit(project, points_world, points_pixel.reshape(-1), packed_params, bounds=bounds, maxfev=20000)
    # 解包所有参数
    params_refined = popt
    intrinsics = params_refined[:5]
    # 相机内参K
    alpha, beta, gamma, u_c, v_c = intrinsics
    K = np.array([[alpha, gamma, u_c],
                  [0., beta, v_c],
                  [0., 0., 1.]])
    k1, k2 = params_refined[5:7]
    # 所有外参
    rt_v = params_refined[7:]
    m = int(len(rt_v) / 5)
    v_rot = []
    v_trans = []
    for i in range(m):
        v_rot.append(rt_v[i * 5:i * 5 + 3])
        v_trans.append(rt_v[i * 5 + 3:(i + 1) * 5])
    v_rot = np.array(v_rot)
    v_trans = np.array(v_trans)
    # 计算重投影误差
    loss_list = []
    for i in range(len(points_world)):
        world = points_world[i].reshape(-1, 3)
        world[:, 2] = 1
        pixel = points_pixel[i].reshape(-1, 2)
        r_m, _ = cv2.Rodrigues(v_rot[i])
        rt_matri = np.eye(3)
        rt_matri[:2, :2] = r_m[:2, :2]
        rt_matri[:2, 2] = v_trans[i].T
        # 投影三维空间点到成像平面上
        y_pre = (rt_matri @ world.T).T
        y_dis = distort([k1, k2], y_pre)
        y_pre = (K @ y_dis.T).T
        y_pre = y_pre[:, :2]
        loss = np.linalg.norm(y_pre[:, :2] - pixel, axis=1)
        loss_list.append(np.mean(loss))
    ret = np.mean(loss_list)
    return ret, K, [k1, k2], v_rot, v_trans
