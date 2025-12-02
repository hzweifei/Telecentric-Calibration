/**
 * @file variation_model.hpp
 * @brief 变差模型类 - 用于图像缺陷检测
 * 
 * 优化说明：
 * 1. 使用OpenCV的Mat替代Python列表，减少内存拷贝
 * 2. 使用OpenMP并行化计算密集型操作
 * 3. 中位数计算使用nth_element算法(O(n))替代完全排序(O(nlogn))
 * 4. 预分配内存，避免频繁分配
 * 5. 使用SIMD优化的OpenCV函数
 */

#ifndef VARIATION_MODEL_HPP
#define VARIATION_MODEL_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

/**
 * @brief 计算模式枚举
 */
enum class CalcMode {
    Standard,  // 使用均值和标准差
    Robust     // 使用中位数和绝对中位差(MAD)
};

/**
 * @brief 读取文件夹下的所有图片路径
 * @param folder_path 文件夹路径
 * @return 图片路径列表
 */
inline std::vector<std::string> readImagesFromFolder(const std::string& folder_path) {
    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF", ".GIF"
    };
    
    std::vector<std::string> images;
    
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        throw std::runtime_error("文件夹不存在: " + folder_path);
    }
    
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // 转小写比较
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            for (const auto& valid_ext : extensions) {
                std::string lower_valid = valid_ext;
                std::transform(lower_valid.begin(), lower_valid.end(), lower_valid.begin(), ::tolower);
                if (ext == lower_valid) {
                    images.push_back(entry.path().string());
                    break;
                }
            }
        }
    }
    
    // 按文件名排序，保证顺序一致
    std::sort(images.begin(), images.end());
    return images;
}

/**
 * @brief 变差模型类
 */
class VariationModel {
public:
    VariationModel() = default;
    ~VariationModel() = default;
    
    /**
     * @brief 读取图像列表
     * @param image_paths 图像路径列表
     */
    void readImages(const std::vector<std::string>& image_paths);
    
    /**
     * @brief 计算变差模型
     * @param mode 计算模式 (Standard: 均值+标准差, Robust: 中位数+MAD)
     */
    void calculateModel(CalcMode mode);
    
    /**
     * @brief 比较测试图像
     * @param test_image 测试图像
     * @param n 容差倍数 (默认1)
     * @param abs_value 绝对容差值 (默认0)
     * @return 缺陷轮廓列表
     */
    std::vector<std::vector<cv::Point>> compareImage(
        const cv::Mat& test_image, 
        double n = 1.0, 
        double abs_value = 0.0) const;
    
    /**
     * @brief 获取均值/中位数图像
     */
    const cv::Mat& getMeanImage() const { return m_image_; }
    
    /**
     * @brief 获取方差/MAD图像
     */
    const cv::Mat& getVarImage() const { return var_image_; }
    
    /**
     * @brief 获取图像数量
     */
    size_t getImageCount() const { return image_count_; }
    
private:
    // 三通道图像堆栈 (num_images x height x width)
    std::vector<cv::Mat> images_blue_;
    std::vector<cv::Mat> images_green_;
    std::vector<cv::Mat> images_red_;
    
    // 均值/中位数图像 (三通道)
    cv::Mat m_image_;
    
    // 方差/MAD图像 (三通道)
    cv::Mat var_image_;
    
    // 图像尺寸
    int height_ = 0;
    int width_ = 0;
    size_t image_count_ = 0;
    
    /**
     * @brief 计算单通道的中位数图像 (优化版)
     * @param channel_images 单通道图像列表
     * @return 中位数图像
     */
    cv::Mat computeMedianOptimized(const std::vector<cv::Mat>& channel_images) const;
    
    /**
     * @brief 计算单通道的MAD图像 (优化版)
     * @param channel_images 单通道图像列表
     * @param median_image 中位数图像
     * @return MAD图像
     */
    cv::Mat computeMADOptimized(const std::vector<cv::Mat>& channel_images, 
                                 const cv::Mat& median_image) const;
};

// ======================== 实现部分 ========================

inline void VariationModel::readImages(const std::vector<std::string>& image_paths) {
    if (image_paths.empty()) {
        throw std::runtime_error("图片路径列表为空");
    }
    
    // 清空之前的数据
    images_blue_.clear();
    images_green_.clear();
    images_red_.clear();
    
    // 预分配空间
    images_blue_.reserve(image_paths.size());
    images_green_.reserve(image_paths.size());
    images_red_.reserve(image_paths.size());
    
    bool first_image = true;
    
    for (const auto& path : image_paths) {
        // 读取图像
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("无法读取文件: " + path);
        }
        
        // 检查图像尺寸
        if (first_image) {
            height_ = image.rows;
            width_ = image.cols;
            first_image = false;
        } else {
            if (image.rows != height_ || image.cols != width_) {
                throw std::runtime_error(
                    "图片大小不一致: " + path + 
                    " 尺寸为 (" + std::to_string(image.rows) + ", " + std::to_string(image.cols) + 
                    "), 参考尺寸为 (" + std::to_string(height_) + ", " + std::to_string(width_) + ")"
                );
            }
        }
        
        // 分离通道并转换为双精度浮点
        std::vector<cv::Mat> channels(3);
        cv::split(image, channels);
        
        cv::Mat blue_f, green_f, red_f;
        channels[0].convertTo(blue_f, CV_64F);
        channels[1].convertTo(green_f, CV_64F);
        channels[2].convertTo(red_f, CV_64F);
        
        images_blue_.push_back(std::move(blue_f));
        images_green_.push_back(std::move(green_f));
        images_red_.push_back(std::move(red_f));
    }
    
    image_count_ = image_paths.size();
}

inline cv::Mat VariationModel::computeMedianOptimized(const std::vector<cv::Mat>& channel_images) const {
    const int n = static_cast<int>(channel_images.size());
    cv::Mat median_image(height_, width_, CV_64F);
    
    // 使用OpenMP并行化行处理
    #pragma omp parallel
    {
        std::vector<double> pixel_values(n);
        
        #pragma omp for schedule(dynamic)
        for (int y = 0; y < height_; ++y) {
            double* median_row = median_image.ptr<double>(y);
            
            for (int x = 0; x < width_; ++x) {
                // 收集所有图像在该像素位置的值
                for (int i = 0; i < n; ++i) {
                    pixel_values[i] = channel_images[i].at<double>(y, x);
                }
                
                // 使用nth_element找中位数 (O(n)复杂度)
                size_t mid = n / 2;
                std::nth_element(pixel_values.begin(), pixel_values.begin() + mid, pixel_values.end());
                
                if (n % 2 == 0) {
                    // 偶数个元素，取中间两个的平均
                    double mid_val = pixel_values[mid];
                    std::nth_element(pixel_values.begin(), pixel_values.begin() + mid - 1, pixel_values.end());
                    median_row[x] = (pixel_values[mid - 1] + mid_val) / 2.0;
                } else {
                    median_row[x] = pixel_values[mid];
                }
            }
        }
    }
    
    return median_image;
}

inline cv::Mat VariationModel::computeMADOptimized(
    const std::vector<cv::Mat>& channel_images, 
    const cv::Mat& median_image) const {
    
    const int n = static_cast<int>(channel_images.size());
    cv::Mat mad_image(height_, width_, CV_64F);
    
    #pragma omp parallel
    {
        std::vector<double> abs_deviations(n);
        
        #pragma omp for schedule(dynamic)
        for (int y = 0; y < height_; ++y) {
            const double* median_row = median_image.ptr<double>(y);
            double* mad_row = mad_image.ptr<double>(y);
            
            for (int x = 0; x < width_; ++x) {
                double median_val = median_row[x];
                
                // 计算绝对偏差
                for (int i = 0; i < n; ++i) {
                    abs_deviations[i] = std::abs(channel_images[i].at<double>(y, x) - median_val);
                }
                
                // 计算绝对偏差的中位数
                size_t mid = n / 2;
                std::nth_element(abs_deviations.begin(), abs_deviations.begin() + mid, abs_deviations.end());
                
                if (n % 2 == 0) {
                    double mid_val = abs_deviations[mid];
                    std::nth_element(abs_deviations.begin(), abs_deviations.begin() + mid - 1, abs_deviations.end());
                    mad_row[x] = (abs_deviations[mid - 1] + mid_val) / 2.0;
                } else {
                    mad_row[x] = abs_deviations[mid];
                }
            }
        }
    }
    
    return mad_image;
}

inline void VariationModel::calculateModel(CalcMode mode) {
    if (images_blue_.empty()) {
        throw std::runtime_error("图像列表为空！");
    }
    
    const int n = static_cast<int>(image_count_);
    
    if (mode == CalcMode::Standard) {
        // ===== Standard模式: 计算均值和标准差 =====
        
        // 初始化累加器
        cv::Mat sum_blue = cv::Mat::zeros(height_, width_, CV_64F);
        cv::Mat sum_green = cv::Mat::zeros(height_, width_, CV_64F);
        cv::Mat sum_red = cv::Mat::zeros(height_, width_, CV_64F);
        cv::Mat sum_sq_blue = cv::Mat::zeros(height_, width_, CV_64F);
        cv::Mat sum_sq_green = cv::Mat::zeros(height_, width_, CV_64F);
        cv::Mat sum_sq_red = cv::Mat::zeros(height_, width_, CV_64F);
        
        // 累加求和
        for (int i = 0; i < n; ++i) {
            cv::add(sum_blue, images_blue_[i], sum_blue);
            cv::add(sum_green, images_green_[i], sum_green);
            cv::add(sum_red, images_red_[i], sum_red);
            
            cv::Mat sq;
            cv::multiply(images_blue_[i], images_blue_[i], sq);
            cv::add(sum_sq_blue, sq, sum_sq_blue);
            
            cv::multiply(images_green_[i], images_green_[i], sq);
            cv::add(sum_sq_green, sq, sum_sq_green);
            
            cv::multiply(images_red_[i], images_red_[i], sq);
            cv::add(sum_sq_red, sq, sum_sq_red);
        }
        
        // 计算均值
        cv::Mat mean_blue, mean_green, mean_red;
        sum_blue /= n;
        sum_green /= n;
        sum_red /= n;
        mean_blue = sum_blue;
        mean_green = sum_green;
        mean_red = sum_red;
        
        // 计算标准差: std = sqrt(E[X^2] - E[X]^2)
        cv::Mat var_blue, var_green, var_red;
        sum_sq_blue /= n;
        sum_sq_green /= n;
        sum_sq_red /= n;
        
        cv::Mat mean_sq;
        cv::multiply(mean_blue, mean_blue, mean_sq);
        cv::subtract(sum_sq_blue, mean_sq, var_blue);
        cv::sqrt(var_blue, var_blue);
        
        cv::multiply(mean_green, mean_green, mean_sq);
        cv::subtract(sum_sq_green, mean_sq, var_green);
        cv::sqrt(var_green, var_green);
        
        cv::multiply(mean_red, mean_red, mean_sq);
        cv::subtract(sum_sq_red, mean_sq, var_red);
        cv::sqrt(var_red, var_red);
        
        // 合并通道
        std::vector<cv::Mat> mean_channels = {mean_blue, mean_green, mean_red};
        std::vector<cv::Mat> var_channels = {var_blue, var_green, var_red};
        cv::merge(mean_channels, m_image_);
        cv::merge(var_channels, var_image_);
        
    } else if (mode == CalcMode::Robust) {
        // ===== Robust模式: 计算中位数和MAD =====
        
        // 并行计算三个通道的中位数
        cv::Mat median_blue, median_green, median_red;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            { median_blue = computeMedianOptimized(images_blue_); }
            
            #pragma omp section
            { median_green = computeMedianOptimized(images_green_); }
            
            #pragma omp section
            { median_red = computeMedianOptimized(images_red_); }
        }
        
        // 合并中位数图像
        std::vector<cv::Mat> median_channels = {median_blue, median_green, median_red};
        cv::merge(median_channels, m_image_);
        
        // 并行计算三个通道的MAD
        cv::Mat mad_blue, mad_green, mad_red;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            { mad_blue = computeMADOptimized(images_blue_, median_blue); }
            
            #pragma omp section
            { mad_green = computeMADOptimized(images_green_, median_green); }
            
            #pragma omp section
            { mad_red = computeMADOptimized(images_red_, median_red); }
        }
        
        // 合并MAD图像
        std::vector<cv::Mat> mad_channels = {mad_blue, mad_green, mad_red};
        cv::merge(mad_channels, var_image_);
    }
}

inline std::vector<std::vector<cv::Point>> VariationModel::compareImage(
    const cv::Mat& test_image, 
    double n, 
    double abs_value) const {
    
    if (m_image_.empty() || var_image_.empty()) {
        throw std::runtime_error("模型未初始化，请先调用calculateModel()");
    }
    
    if (test_image.rows != height_ || test_image.cols != width_) {
        throw std::runtime_error("测试图像尺寸与模型不匹配");
    }
    
    // 分离测试图像通道
    std::vector<cv::Mat> test_channels(3);
    cv::split(test_image, test_channels);
    
    cv::Mat blue, green, red;
    test_channels[0].convertTo(blue, CV_64F);
    test_channels[1].convertTo(green, CV_64F);
    test_channels[2].convertTo(red, CV_64F);
    
    // 分离模型通道
    std::vector<cv::Mat> m_channels(3), var_channels(3);
    cv::split(m_image_, m_channels);
    cv::split(var_image_, var_channels);
    
    // 计算容差范围
    cv::Mat tolerance_blue, tolerance_green, tolerance_red;
    cv::Mat scaled_var;
    
    // Blue channel
    var_channels[0].copyTo(scaled_var);
    scaled_var *= n;
    scaled_var += abs_value;
    
    cv::Mat blue_min = m_channels[0] - scaled_var;
    cv::Mat blue_max = m_channels[0] + scaled_var;
    
    // Green channel
    var_channels[1].copyTo(scaled_var);
    scaled_var *= n;
    scaled_var += abs_value;
    
    cv::Mat green_min = m_channels[1] - scaled_var;
    cv::Mat green_max = m_channels[1] + scaled_var;
    
    // Red channel
    var_channels[2].copyTo(scaled_var);
    scaled_var *= n;
    scaled_var += abs_value;
    
    cv::Mat red_min = m_channels[2] - scaled_var;
    cv::Mat red_max = m_channels[2] + scaled_var;
    
    // 创建掩膜 (超出范围的像素)
    cv::Mat blue_mask_low, blue_mask_high, blue_mask;
    cv::compare(blue, blue_min, blue_mask_low, cv::CMP_LT);
    cv::compare(blue, blue_max, blue_mask_high, cv::CMP_GT);
    cv::bitwise_or(blue_mask_low, blue_mask_high, blue_mask);
    
    cv::Mat red_mask_low, red_mask_high, red_mask;
    cv::compare(red, red_min, red_mask_low, cv::CMP_LT);
    cv::compare(red, red_max, red_mask_high, cv::CMP_GT);
    cv::bitwise_or(red_mask_low, red_mask_high, red_mask);
    
    // 合并掩膜 (与Python代码一致，只使用蓝色和红色通道)
    cv::Mat combined_mask;
    cv::bitwise_or(blue_mask, red_mask, combined_mask);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combined_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 过滤小面积轮廓
    std::vector<std::vector<cv::Point>> filtered_contours;
    filtered_contours.reserve(contours.size());
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 5.0) {
            filtered_contours.push_back(contour);
        }
    }
    
    return filtered_contours;
}

#endif // VARIATION_MODEL_HPP
