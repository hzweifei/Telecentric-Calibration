/**
 * @file main.cpp
 * @brief 变差模型使用示例
 */

#include "variation_model.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    try {
        std::string folder_path;
        std::string test_image_path;
        std::string mode_str = "Standard";
        double n = 3.0;
        double abs_value = 10.0;
        
        // 解析命令行参数
        if (argc >= 2) {
            folder_path = argv[1];
        } else {
            std::cout << "用法: " << argv[0] << " <图片文件夹> [测试图片] [模式:Standard/Robust] [n倍数] [绝对容差]" << std::endl;
            std::cout << "示例: " << argv[0] << " ./images test.bmp Robust 3.0 10.0" << std::endl;
            return 1;
        }
        
        if (argc >= 3) test_image_path = argv[2];
        if (argc >= 4) mode_str = argv[3];
        if (argc >= 5) n = std::stod(argv[4]);
        if (argc >= 6) abs_value = std::stod(argv[5]);
        
        // 解析模式
        CalcMode mode = (mode_str == "Robust") ? CalcMode::Robust : CalcMode::Standard;
        
        // 读取图片列表
        std::cout << "正在读取文件夹: " << folder_path << std::endl;
        auto image_paths = readImagesFromFolder(folder_path);
        std::cout << "找到 " << image_paths.size() << " 张图片" << std::endl;
        
        if (image_paths.empty()) {
            std::cerr << "错误: 文件夹中没有找到图片" << std::endl;
            return 1;
        }
        
        // 创建变差模型
        VariationModel model;
        
        // 读取图像
        auto start = std::chrono::high_resolution_clock::now();
        model.readImages(image_paths);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "读取图像耗时: " << duration.count() << " ms" << std::endl;
        
        // 计算模型
        std::cout << "使用模式: " << mode_str << std::endl;
        start = std::chrono::high_resolution_clock::now();
        model.calculateModel(mode);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "计算模型耗时: " << duration.count() << " ms" << std::endl;
        
        // 保存均值/中位数图像
        cv::Mat mean_img_8u;
        model.getMeanImage().convertTo(mean_img_8u, CV_8U);
        cv::imwrite("mean_image.bmp", mean_img_8u);
        std::cout << "已保存均值/中位数图像: mean_image.bmp" << std::endl;
        
        // 保存方差/MAD图像 (归一化显示)
        cv::Mat var_img;
        cv::normalize(model.getVarImage(), var_img, 0, 255, cv::NORM_MINMAX);
        var_img.convertTo(var_img, CV_8U);
        cv::imwrite("var_image.bmp", var_img);
        std::cout << "已保存方差/MAD图像: var_image.bmp" << std::endl;
        
        // 如果提供了测试图像，进行比较
        if (!test_image_path.empty()) {
            cv::Mat test_image = cv::imread(test_image_path, cv::IMREAD_COLOR);
            if (test_image.empty()) {
                std::cerr << "错误: 无法读取测试图像: " << test_image_path << std::endl;
                return 1;
            }
            
            std::cout << "\n正在比较测试图像: " << test_image_path << std::endl;
            std::cout << "参数: n=" << n << ", abs_value=" << abs_value << std::endl;
            
            start = std::chrono::high_resolution_clock::now();
            auto contours = model.compareImage(test_image, n, abs_value);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "比较图像耗时: " << duration.count() << " ms" << std::endl;
            
            std::cout << "检测到 " << contours.size() << " 个缺陷区域" << std::endl;
            
            // 在测试图像上绘制缺陷轮廓
            cv::Mat result = test_image.clone();
            cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255), 2);
            cv::imwrite("result.bmp", result);
            std::cout << "已保存结果图像: result.bmp" << std::endl;
            
            // 输出每个缺陷的面积
            for (size_t i = 0; i < contours.size(); ++i) {
                double area = cv::contourArea(contours[i]);
                cv::Rect bbox = cv::boundingRect(contours[i]);
                std::cout << "缺陷 " << (i + 1) << ": 面积=" << area 
                          << ", 位置=(" << bbox.x << ", " << bbox.y << ")"
                          << ", 尺寸=" << bbox.width << "x" << bbox.height << std::endl;
            }
        }
        
        std::cout << "\n处理完成!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}
