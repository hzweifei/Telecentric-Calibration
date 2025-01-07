import numpy as np
import cv2
from .edgepixel import EdgePixel
from .edges_iterN import h_edges, v_edges


def main_iterN(F, threshold, iters, order):
    ep = EdgePixel()
    rows, cols = np.shape(F)

        
    I = np.zeros((rows, cols))
    C = np.zeros((rows, cols))
    
    # smooth image
    [x, y] = np.meshgrid(np.arange(cols), np.arange(rows))
    w = 0.75
    G = np.copy(F)
    G[1:rows - 1, 1:cols - 1] = (F[0:rows - 2, 0:cols - 2] + F[0:rows - 2, 1:cols - 1] + F[0:rows - 2, 2:cols] +
                                 F[1:rows - 1, 0:cols - 2] + F[1:rows - 1, 1:cols - 1] + F[1:rows - 1, 2:cols] +
                                 F[2:rows, 0:cols - 2] + F[2:rows, 1:cols - 1] + F[2:rows, 2:cols]) / 9

    # use sobel
    Gx = cv2.Sobel(G, cv2.CV_64F, 1, 0, ksize=3)  # x 方向梯度
    Gy = cv2.Sobel(G, cv2.CV_64F, 0, 1, ksize=3)  # y 方向梯度
    # 计算幅值
    grad = cv2.magnitude(Gx, Gy)

    # detect edge pixels with maximum Gy (not including margins)
    absGyInner = np.abs(Gy[5:rows - 5, 2: cols - 2])
    absGxInner = np.abs(Gx[2:rows - 2, 5: cols - 5])

    Ey = np.zeros((rows, cols), dtype=np.bool_)
    Ex = np.zeros((rows, cols), dtype=np.bool_)

    Ey[5: rows - 5, 2: cols - 2] = np.logical_and.reduce([
        grad[5: rows - 5, 2: cols - 2] > threshold,
        absGyInner >= np.abs(Gx[5: rows - 5, 2: cols - 2]),
        absGyInner >= np.abs(Gy[4: rows - 6, 2: cols - 2]),
        absGyInner > np.abs(Gy[6: rows - 4, 2: cols - 2])
    ])

    Ex[2: rows - 2, 5: cols - 5] = np.logical_and.reduce([
        grad[2: rows - 2, 5: cols - 5] > threshold,
        absGxInner > np.abs(Gy[2: rows - 2, 5: cols - 5]),
        absGxInner >= np.abs(Gx[2: rows - 2, 4: cols - 6]),
        absGxInner > np.abs(Gx[2: rows - 2, 6: cols - 4])
    ])

    Ey = Ey.ravel('F')
    Ex = Ex.ravel('F')
    y = y.ravel('F')
    x = x.ravel('F')

    edges_y = (x[Ey] * rows + y[Ey])
    edges_x = (x[Ex] * rows + y[Ex])

    Gx = Gx.ravel('F')
    Gy = Gy.ravel('F')

    x_y, y_y, edges_y, nx_y, ny_y, i0_y, i1_y, curv_y, I, C, G = h_edges(F, G, rows, Gx, Gy, w, edges_y, order,
                                                                         threshold, x, y, cols, I, C)
    x_x, y_x, edges_x, nx_x, ny_x, i0_x, i1_x, curv_x, I, C, G = v_edges(F, G, rows, Gx, Gy, w, edges_x, order,
                                                                          threshold, x, y, I, C)

    # compute final subimage
    I[C > 0] = I[C > 0] / C[C > 0]
    I[C == 0] = G[C == 0]

    # save results
    ep.ny = np.concatenate((ny_y, ny_x), axis=0)
    ep.nx = np.concatenate((nx_y, nx_x), axis=0)

    ep.y = np.concatenate((y_y, y_x), axis=0)
    ep.x = np.concatenate((x_y, x_x), axis=0)

    ep.position = np.concatenate((edges_y, edges_x), axis=0)
    ep.curv = np.concatenate((curv_y, curv_x), axis=0)
    ep.i0 = np.concatenate((i0_y, i0_x), axis=0)
    ep.i1 = np.concatenate((i1_y, i1_x), axis=0)
    ep.sub_position = np.column_stack((ep.x, ep.y)).astype(np.float32)

    return ep, I
