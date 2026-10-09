#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "object_detector.hpp"

struct TrackedObject {
    int id;
    cv::Rect box;
    int class_id;
    float confidence;
    int frames_since_seen;
};

class Tracker {
public:
    Tracker(float iou_threshold = 0.3f, int max_missed_frames = 5);

    std::vector<TrackedObject> update(const std::vector<Detection>& detections);

private:
    float iou_threshold_;
    int max_missed_frames_;
    int next_id_;
    std::vector<TrackedObject> tracked_objects_;

    float computeIoU(const cv::Rect& a, const cv::Rect& b);
};