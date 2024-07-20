from enum import Enum


class PatternType(Enum):
    CHESSBOARD = 0  # 棋盘格
    CIRCLE = 1  # 圆点
    RING = 2  # 圆环
    HALCON_CIRCLE = 3  # halcon圆点标定板


class PatternInfo:
    def __init__(self, pattern_type: PatternType, shape: tuple, distance, pixel_size: tuple, radius_inside=None,
                 radius_outside=None):
        """
        :param pattern_type: PatternType标定图案样式
        :param shape: (x,y)控制点形状
        :param distance: 控制点间距(单位：mm)
        :param pixel_size: (x,y) 显示器像素尺寸，用于将真实尺寸转换为像素尺寸(单位：um)
        :param radius_inside: 圆环内径,对于圆点图形：圆点半径(单位：mm)
        :param radius_outside: 圆环外径，(单位：mm)
        """
        self.type = pattern_type  # 标定图案样式
        self.shape = shape  # 控制点形状
        self.distance = distance  # 控制点间距(单位：mm)
        self.pixel_size = pixel_size  # 显示器像素尺寸，用于将真实尺寸转换为像素尺寸(单位：um)

        if pattern_type == PatternType.CIRCLE:
            if radius_outside is None:
                raise ValueError("radius_outside is required for CIRCLE pattern type")
            self.radius_outside = None
            self.radius_inside = radius_inside  # 圆点半径(单位：mm)
        elif pattern_type == PatternType.RING:
            if radius_inside is None or radius_outside is None:
                raise ValueError("Both radius_inside and radius_outside are required for RING pattern type")
            self.radius_inside = radius_inside  # 圆环内径(单位：mm)
            self.radius_outside = radius_outside  # 圆环外径(单位：mm)
        elif pattern_type == PatternType.CHESSBOARD:
            pass
        elif pattern_type == PatternType.HALCON_CIRCLE:
            pass
        else:
            raise ValueError("Unsupported pattern type")
