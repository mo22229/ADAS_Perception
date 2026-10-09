#include "lane_detector.hpp"
#include <algorithm>
#include <cmath>

const std::vector<int> LaneDetector::ROW_ANCHOR = {
    121, 131, 141, 150, 160, 170, 180, 189, 199,
    209, 219, 228, 238, 248, 258, 267, 277, 287
};

LaneDetector::LaneDetector(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ADAS_Lane"),
      session_(nullptr)
{
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session_.GetOutputNameAllocated(0, allocator);

    input_name_ = input_name_ptr.get();
    output_name_ = output_name_ptr.get();

    input_names_[0] = input_name_.c_str();
    output_names_[0] = output_name_.c_str();
}

std::vector<Lane> LaneDetector::detect(const cv::Mat& frame)
{
    int orig_w = frame.cols;
    int orig_h = frame.rows;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> input_tensor_values(1 * 3 * INPUT_H * INPUT_W);
    int plane_size = INPUT_H * INPUT_W;
    for (int y = 0; y < INPUT_H; y++) {
        for (int x = 0; x < INPUT_W; x++) {
            cv::Vec3f pixel = rgb.at<cv::Vec3f>(y, x);
            for (int c = 0; c < 3; c++) {
                float val = (pixel[c] - mean[c]) / std_[c];
                input_tensor_values[c * plane_size + y * INPUT_W + x] = val;
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
        input_names_, &input_tensor, 1,
        output_names_, 1);

    float* out = output_tensors[0].GetTensorMutableData<float>();

    const int G = GRIDING_NUM + 1;
    const int R = CLS_NUM_PER_LANE;
    const int L = NUM_LANES;

    auto at = [&](int g, int row, int lane) -> float {
        int flipped_row = R - 1 - row;
        return out[g * R * L + flipped_row * L + lane];
    };

    float col_sample_w = (float)(INPUT_W - 1) / (GRIDING_NUM - 1);

    std::vector<Lane> lanes(L);

    for (int lane = 0; lane < L; lane++) {
        std::vector<float> loc(R, 0.0f);
        int valid_count = 0;

        for (int row = 0; row < R; row++) {
            float max_val = -1e9f;
            for (int g = 0; g < GRIDING_NUM; g++) {
                max_val = std::max(max_val, at(g, row, lane));
            }
            float sum_exp = 0.0f;
            std::vector<float> exp_vals(GRIDING_NUM);
            for (int g = 0; g < GRIDING_NUM; g++) {
                exp_vals[g] = std::exp(at(g, row, lane) - max_val);
                sum_exp += exp_vals[g];
            }

            float expectation = 0.0f;
            for (int g = 0; g < GRIDING_NUM; g++) {
                float prob = exp_vals[g] / sum_exp;
                expectation += prob * (float)(g + 1);
            }

            int best_g = 0;
            float best_val = at(0, row, lane);
            for (int g = 1; g < G; g++) {
                float v = at(g, row, lane);
                if (v > best_val) {
                    best_val = v;
                    best_g = g;
                }
            }

            if (best_g == GRIDING_NUM) {
                loc[row] = 0.0f;
            } else {
                loc[row] = expectation;
                valid_count++;
            }
        }

        if (valid_count > 2) {
            for (int row = 0; row < R; row++) {
                if (loc[row] > 0.0f) {
                    int x = (int)(loc[row] * col_sample_w * orig_w / INPUT_W) - 1;
                    int y = (int)(orig_h * ((float)ROW_ANCHOR[R - 1 - row] / 288.0f)) - 1;
                    lanes[lane].points.push_back(cv::Point(x, y));
                }
            }
        }
    }

    std::vector<Lane> result;
    for (const auto& l : lanes) {
        if (!l.points.empty()) result.push_back(l);
    }
    return result;
}