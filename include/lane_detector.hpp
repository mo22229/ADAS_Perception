#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Lane {
    std::vector<cv::Point> points;
};

class LaneDetector {
public:
    LaneDetector(const std::string& model_path);

    std::vector<Lane> detect(const cv::Mat& frame);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    std::string input_name_;
    std::string output_name_;
    const char* input_names_[1];
    const char* output_names_[1];

    static const int INPUT_W = 800;
    static const int INPUT_H = 288;
    static const int GRIDING_NUM = 200;
    static const int CLS_NUM_PER_LANE = 18;
    static const int NUM_LANES = 4;

    static const std::vector<int> ROW_ANCHOR;
};