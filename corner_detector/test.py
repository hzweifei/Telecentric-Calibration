from modules import *
import high_accuracy_corner_detector
from Pattern_type import *

if __name__ == '__main__':
    print("测试文件")
    # chessboard test_______________________________________________________________________
    image = cv2.imread("test_image/checkboard.BMP")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()
    # 寻找角点
    pattern_info = PatternInfo(PatternType.CHESSBOARD, (19, 17), 0.5, (4.8, 4.8))
    ret, corner_points = high_accuracy_corner_detector.find_corners(gray, pattern_info)
    # 显示
    w, h = pattern_info.shape
    cv2.drawChessboardCorners(image, (w, h), corner_points, ret)
    plt.imshow(image)
    plt.show()

    # chessboard test_______________________________________________________________________
