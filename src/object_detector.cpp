#include "object_detector.hpp"
#include <algorithm>

const std::vector<std::string> ObjectDetector::CLASS_NAMES = {
    "Car", "Van", "Truck", "Pedestrian",
    "Person_sitting", "Cyclist", "Tram", "Misc"
};

const std::vector<float> ObjectDetector::CLASS_THRESHOLDS = {
    0.25f,  // Car
    0.40f,  // Van
    0.40f,  // Truck
    0.25f,  // Pedestrian
    0.70f,  // Person_sitting
    0.40f,  // Cyclist
    0.70f,  // Tram
    0.70f   // Misc
};

ObjectDetector::ObjectDetector(const std::string& model_path, int input_size,
                                 float conf_threshold, float nms_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ADAS"),
      session_(nullptr),
      input_size_(input_size),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold)
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

std::vector<Detection> ObjectDetector::detect(const cv::Mat& frame)
{
    int orig_width = frame.cols;
    int orig_height = frame.rows;

    float scale = std::min((float)input_size_ / orig_width, (float)input_size_ / orig_height);
    int new_width = (int)(orig_width * scale);
    int new_height = (int)(orig_height * scale);
    int pad_x = (input_size_ - new_width) / 2;
    int pad_y = (input_size_ - new_height) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));

    cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_width, new_height)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values(1 * 3 * input_size_ * input_size_);
    int plane_size = input_size_ * input_size_;
    for (int y = 0; y < input_size_; y++) {
        for (int x = 0; x < input_size_; x++) {
            cv::Vec3f pixel = rgb.at<cv::Vec3f>(y, x);
            input_tensor_values[0 * plane_size + y * input_size_ + x] = pixel[0];
            input_tensor_values[1 * plane_size + y * input_size_ + x] = pixel[1];
            input_tensor_values[2 * plane_size + y * input_size_ + x] = pixel[2];
        }
    }

    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
        input_names_, &input_tensor, 1,
        output_names_, 1);

    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_boxes = output_shape[2];
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_boxes; i++) {
        float cx = output_data[0 * num_boxes + i];
        float cy = output_data[1 * num_boxes + i];
        float w  = output_data[2 * num_boxes + i];
        float h  = output_data[3 * num_boxes + i];

        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = output_data[(4 + c) * num_boxes + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < CLASS_THRESHOLDS[best_class]) continue;

        int left = (int)((cx - w / 2 - pad_x) / scale);
        int top = (int)((cy - h / 2 - pad_y) / scale);
        int box_w = (int)(w / scale);
        int box_h = (int)(h / scale);

        boxes.push_back(cv::Rect(left, top, box_w, box_h));
        confidences.push_back(best_score);
        class_ids.push_back(best_class);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    std::vector<Detection> results;
    for (int idx : indices) {
        Detection d;
        d.box = boxes[idx];
        d.confidence = confidences[idx];
        d.class_id = class_ids[idx];
        results.push_back(d);
    }

    return results;
}