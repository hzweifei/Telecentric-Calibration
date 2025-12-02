/**
 * @file variation_model_simd.hpp
 * @brief 变差模型类 - 高性能SIMD优化版本
 * 
 * 额外优化：
 * 1. 使用连续内存布局提高缓存命中率
 * 2. 批量中位数计算优化
 * 3. 使用OpenCV的parallel_for_实现自动并行化
 */

#ifndef VARIATION_MODEL_SIMD_HPP
#define VARIATION_MODEL_SIMD_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/parallel/parallel_backend.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <numeric>

namespace fs = std::filesystem;

/**
 * @brief 计算模式枚举
 */
enum class CalcModeSIMD {
    Standard,
    Robust
};

/**
 * @brief 高性能变差模型类
 * 
 * 主要优化：使用 (height*width, num_images) 的内存布局
 * 这样每个像素位置的所有图像值是连续的，有利于中位数计算
 */
class VariationModelSIMD {
public:
    VariationModelSIMD() = default;
    
    /**
     * @brief 读取图像列表（优化内存布局）
     */
    void readImages(const std::vector<std::string>& image_paths) {
        if (image_paths.empty()) {
            throw std::runtime_error("图片路径列表为空");
        }
        
        // 首先读取第一张图像获取尺寸
        cv::Mat first_img = cv::imread(image_paths[0], cv::IMREAD_COLOR);
        if (first_img.empty()) {
            throw std::runtime_error("无法读取文件: " + image_paths[0]);
        }
        
        height_ = first_img.rows;
        width_ = first_img.cols;
        num_images_ = image_paths.size();
        num_pixels_ = height_ * width_;
        
        // 使用优化的内存布局: (num_pixels, num_images)
        // 这样每个像素位置的所有值是连续的
        data_blue_.resize(num_pixels_ * num_images_);
        data_green_.resize(num_pixels_ * num_images_);
        data_red_.resize(num_pixels_ * num_images_);
        
        // 并行读取和转换图像
        cv::parallel_for_(cv::Range(0, num_images_), [&](const cv::Range& range) {
            for (int idx = range.start; idx < range.end; ++idx) {
                cv::Mat img = cv::imread(image_paths[idx], cv::IMREAD_COLOR);
                if (img.empty()) {
                    throw std::runtime_error("无法读取文件: " + image_paths[idx]);
                }
                
                if (img.rows != height_ || img.cols != width_) {
                    throw std::runtime_error("图片大小不一致: " + image_paths[idx]);
                }
                
                // 分离通道并存储到优化布局
                std::vector<cv::Mat> channels(3);
                cv::split(img, channels);
                
                for (int y = 0; y < height_; ++y) {
                    const uchar* b_row = channels[0].ptr<uchar>(y);
                    const uchar* g_row = channels[1].ptr<uchar>(y);
                    const uchar* r_row = channels[2].ptr<uchar>(y);
                    
                    for (int x = 0; x < width_; ++x) {
                        size_t pixel_idx = y * width_ + x;
                        size_t offset = pixel_idx * num_images_ + idx;
                        data_blue_[offset] = static_cast<float>(b_row[x]);
                        data_green_[offset] = static_cast<float>(g_row[x]);
                        data_red_[offset] = static_cast<float>(r_row[x]);
                    }
                }
            }
        });
    }
    
    /**
     * @brief 计算变差模型（使用优化内存布局）
     */
    void calculateModel(CalcModeSIMD mode) {
        if (data_blue_.empty()) {
            throw std::runtime_error("图像列表为空！");
        }
        
        // 分配结果内存
        m_image_.create(height_, width_, CV_32FC3);
        var_image_.create(height_, width_, CV_32FC3);
        
        if (mode == CalcModeSIMD::Standard) {
            calculateStandard();
        } else {
            calculateRobust();
        }
    }
    
    /**
     * @brief 比较测试图像
     */
    std::vector<std::vector<cv::Point>> compareImage(
        const cv::Mat& test_image,
        float n = 1.0f,
        float abs_value = 0.0f) const {
        
        if (m_image_.empty() || var_image_.empty()) {
            throw std::runtime_error("模型未初始化");
        }
        
        // 转换测试图像为浮点
        cv::Mat test_float;
        test_image.convertTo(test_float, CV_32FC3);
        
        // 计算容差范围
        cv::Mat tolerance = var_image_ * n + abs_value;
        cv::Mat min_img = m_image_ - tolerance;
        cv::Mat max_img = m_image_ + tolerance;
        
        // 创建掩膜
        cv::Mat mask = cv::Mat::zeros(height_, width_, CV_8U);
        
        cv::parallel_for_(cv::Range(0, height_), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const cv::Vec3f* test_row = test_float.ptr<cv::Vec3f>(y);
                const cv::Vec3f* min_row = min_img.ptr<cv::Vec3f>(y);
                const cv::Vec3f* max_row = max_img.ptr<cv::Vec3f>(y);
                uchar* mask_row = mask.ptr<uchar>(y);
                
                for (int x = 0; x < width_; ++x) {
                    const cv::Vec3f& pixel = test_row[x];
                    const cv::Vec3f& min_val = min_row[x];
                    const cv::Vec3f& max_val = max_row[x];
                    
                    // 检查蓝色和红色通道（与Python代码一致）
                    bool out_of_range = 
                        (pixel[0] < min_val[0] || pixel[0] > max_val[0]) ||  // Blue
                        (pixel[2] < min_val[2] || pixel[2] > max_val[2]);    // Red
                    
                    mask_row[x] = out_of_range ? 255 : 0;
                }
            }
        });
        
        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        // 过滤小面积
        std::vector<std::vector<cv::Point>> result;
        for (auto& contour : contours) {
            if (cv::contourArea(contour) > 5.0) {
                result.push_back(std::move(contour));
            }
        }
        
        return result;
    }
    
    const cv::Mat& getMeanImage() const { return m_image_; }
    const cv::Mat& getVarImage() const { return var_image_; }
    
private:
    std::vector<float> data_blue_;
    std::vector<float> data_green_;
    std::vector<float> data_red_;
    
    cv::Mat m_image_;
    cv::Mat var_image_;
    
    int height_ = 0;
    int width_ = 0;
    size_t num_images_ = 0;
    size_t num_pixels_ = 0;
    
    /**
     * @brief Standard模式计算（均值+标准差）
     */
    void calculateStandard() {
        cv::parallel_for_(cv::Range(0, height_), [this](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                cv::Vec3f* m_row = m_image_.ptr<cv::Vec3f>(y);
                cv::Vec3f* v_row = var_image_.ptr<cv::Vec3f>(y);
                
                for (int x = 0; x < width_; ++x) {
                    size_t pixel_idx = y * width_ + x;
                    size_t base_offset = pixel_idx * num_images_;
                    
                    // 计算三个通道的均值和标准差
                    for (int c = 0; c < 3; ++c) {
                        const float* data = (c == 0) ? &data_blue_[base_offset] :
                                            (c == 1) ? &data_green_[base_offset] :
                                                       &data_red_[base_offset];
                        
                        // 使用Welford在线算法计算均值和方差
                        double mean = 0.0;
                        double M2 = 0.0;
                        
                        for (size_t i = 0; i < num_images_; ++i) {
                            double delta = data[i] - mean;
                            mean += delta / (i + 1);
                            double delta2 = data[i] - mean;
                            M2 += delta * delta2;
                        }
                        
                        m_row[x][c] = static_cast<float>(mean);
                        v_row[x][c] = static_cast<float>(std::sqrt(M2 / num_images_));
                    }
                }
            }
        });
    }
    
    /**
     * @brief Robust模式计算（中位数+MAD）
     */
    void calculateRobust() {
        // 为每个线程分配临时缓冲区
        int num_threads = cv::getNumThreads();
        std::vector<std::vector<float>> temp_buffers(num_threads);
        for (auto& buf : temp_buffers) {
            buf.resize(num_images_);
        }
        
        cv::parallel_for_(cv::Range(0, height_), [this, &temp_buffers](const cv::Range& range) {
            int thread_id = cv::getThreadNum();
            std::vector<float>& buffer = temp_buffers[thread_id];
            
            for (int y = range.start; y < range.end; ++y) {
                cv::Vec3f* m_row = m_image_.ptr<cv::Vec3f>(y);
                cv::Vec3f* v_row = var_image_.ptr<cv::Vec3f>(y);
                
                for (int x = 0; x < width_; ++x) {
                    size_t pixel_idx = y * width_ + x;
                    size_t base_offset = pixel_idx * num_images_;
                    
                    // 三个通道
                    for (int c = 0; c < 3; ++c) {
                        const float* data = (c == 0) ? &data_blue_[base_offset] :
                                            (c == 1) ? &data_green_[base_offset] :
                                                       &data_red_[base_offset];
                        
                        // 复制数据到临时缓冲区
                        std::copy(data, data + num_images_, buffer.begin());
                        
                        // 计算中位数
                        float median = computeMedian(buffer);
                        m_row[x][c] = median;
                        
                        // 计算绝对偏差
                        for (size_t i = 0; i < num_images_; ++i) {
                            buffer[i] = std::abs(buffer[i] - median);
                        }
                        
                        // 计算MAD
                        v_row[x][c] = computeMedian(buffer);
                    }
                }
            }
        });
    }
    
    /**
     * @brief 快速中位数计算（使用nth_element）
     */
    static float computeMedian(std::vector<float>& data) {
        size_t n = data.size();
        size_t mid = n / 2;
        
        std::nth_element(data.begin(), data.begin() + mid, data.end());
        
        if (n % 2 == 0) {
            float mid_val = data[mid];
            std::nth_element(data.begin(), data.begin() + mid - 1, data.end());
            return (data[mid - 1] + mid_val) / 2.0f;
        }
        
        return data[mid];
    }
};

#endif // VARIATION_MODEL_SIMD_HPP
