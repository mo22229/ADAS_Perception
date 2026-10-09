#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class ObjectDetector {
public:
    ObjectDetector(const std::string& model_path, int input_size = 640,
                   float conf_threshold = 0.25f, float nms_threshold = 0.45f);

    std::vector<Detection> detect(const cv::Mat& frame);

    static const std::vector<std::string> CLASS_NAMES;
    static const std::vector<float> CLASS_THRESHOLDS;

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    std::string input_name_;
    std::string output_name_;
    const char* input_names_[1];
    const char* output_names_[1];

    int input_size_;
    float conf_threshold_;
    float nms_threshold_;
    static const int NUM_CLASSES = 8;
};