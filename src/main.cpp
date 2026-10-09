#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "object_detector.hpp"
#include "tracker.hpp"
#include "lane_detector.hpp"

int main()
{
    const std::string object_model_path = "../models/object_detection/best.onnx";
    const std::string lane_model_path = "../models/lane_detection/model.onnx";
    const std::string video_path = "../data/nD_16.mp4";

    ObjectDetector detector(object_model_path);
    Tracker tracker;
    LaneDetector lane_detector(lane_model_path);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << video_path << std::endl;
        return -1;
    }

    double video_fps = cap.get(cv::CAP_PROP_FPS);
    int frame_time_ms = (int)(1000.0 / video_fps);
    std::cout << "Video FPS: " << video_fps << std::endl;

    long total_processing_ms = 0;
    int frame_count = 0;

    const cv::Scalar lane_colors[4] = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255),
        cv::Scalar(0, 165, 255), cv::Scalar(255, 0, 255)
    };

    cv::Mat frame;
    while (cap.read(frame)) {

        auto frame_start = std::chrono::steady_clock::now();

        std::vector<Detection> detections = detector.detect(frame);
        std::vector<TrackedObject> tracked = tracker.update(detections);
        std::vector<Lane> lanes = lane_detector.detect(frame);
        std::cout << "Lanes detected: " << lanes.size() << std::endl;

        for (size_t i = 0; i < lanes.size(); i++) {
            for (size_t p = 0; p + 1 < lanes[i].points.size(); p++) {
                cv::line(frame, lanes[i].points[p], lanes[i].points[p + 1],
                          lane_colors[i % 4], 4);
            }
        }

        for (const auto& obj : tracked) {
            cv::rectangle(frame, obj.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID" + std::to_string(obj.id) + " " +
                                 ObjectDetector::CLASS_NAMES[obj.class_id] + " " +
                                 std::to_string((int)(obj.confidence * 100)) + "%";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(frame,
                cv::Point(obj.box.x, obj.box.y - text_size.height - 6),
                cv::Point(obj.box.x + text_size.width + 4, obj.box.y),
                cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(obj.box.x + 2, obj.box.y - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        auto frame_end = std::chrono::steady_clock::now();
        int processing_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_end - frame_start).count();

        total_processing_ms += processing_ms;
        frame_count++;

        double actual_fps = processing_ms > 0 ? 1000.0 / processing_ms : 0.0;
        std::string fps_text = "Processing FPS: " + std::to_string((int)actual_fps);
        cv::putText(frame, fps_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::namedWindow("ADAS Live Stream", cv::WINDOW_NORMAL);
        cv::resizeWindow("ADAS Live Stream", 960, 540);
        cv::imshow("ADAS Live Stream", frame);

        int wait_time = frame_time_ms - processing_ms;
        if (wait_time < 1) wait_time = 1;

        if (cv::waitKey(wait_time) == 'q') break;
    }

    cap.release();

    if (frame_count > 0) {
        double avg_ms = (double)total_processing_ms / frame_count;
        double avg_fps = 1000.0 / avg_ms;
        std::cout << "----------------------------------" << std::endl;
        std::cout << "Frames processed: " << frame_count << std::endl;
        std::cout << "Average processing time: " << avg_ms << " ms/frame" << std::endl;
        std::cout << "Average FPS: " << avg_fps << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }

    std::cout << "Stream finished." << std::endl;
    return 0;
}