#include "tracker.hpp"

Tracker::Tracker(float iou_threshold, int max_missed_frames)
    : iou_threshold_(iou_threshold),
      max_missed_frames_(max_missed_frames),
      next_id_(0)
{
}

float Tracker::computeIoU(const cv::Rect& a, const cv::Rect& b)
{
    int intersection_area = (a & b).area();
    int union_area = a.area() + b.area() - intersection_area;
    if (union_area == 0) return 0.0f;
    return (float)intersection_area / (float)union_area;
}

std::vector<TrackedObject> Tracker::update(const std::vector<Detection>& detections)
{
    std::vector<bool> detection_matched(detections.size(), false);
    std::vector<bool> track_matched(tracked_objects_.size(), false);

    // الخطوة 1: نحاول نلاقي أحسن تطابق بين كل مربع قديم وكل مربع جديد
    for (size_t t = 0; t < tracked_objects_.size(); t++) {
        float best_iou = iou_threshold_;
        int best_match = -1;

        for (size_t d = 0; d < detections.size(); d++) {
            if (detection_matched[d]) continue;

            float iou = computeIoU(tracked_objects_[t].box, detections[d].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_match = (int)d;
            }
        }

        if (best_match != -1) {
            // لقينا تطابق: نحدّث مكان العربية القديمة بالمكان الجديد، ونحافظ على نفس الـID
            tracked_objects_[t].box = detections[best_match].box;
            tracked_objects_[t].class_id = detections[best_match].class_id;
            tracked_objects_[t].confidence = detections[best_match].confidence;
            tracked_objects_[t].frames_since_seen = 0;
            detection_matched[best_match] = true;
            track_matched[t] = true;
        } else {
            // مفيش تطابق: نزود عداد الغياب
            tracked_objects_[t].frames_since_seen++;
        }
    }

    // الخطوة 2: أي مربع جديد معملوش match، يبقى عربية جديدة، نديها ID جديد
    for (size_t d = 0; d < detections.size(); d++) {
        if (!detection_matched[d]) {
            TrackedObject new_obj;
            new_obj.id = next_id_++;
            new_obj.box = detections[d].box;
            new_obj.class_id = detections[d].class_id;
            new_obj.confidence = detections[d].confidence;
            new_obj.frames_since_seen = 0;
            tracked_objects_.push_back(new_obj);
        }
    }

    // الخطوة 3: نشيل أي عربية اختفت لمدة أطول من المسموح
    std::vector<TrackedObject> remaining;
    for (const auto& obj : tracked_objects_) {
        if (obj.frames_since_seen <= max_missed_frames_) {
            remaining.push_back(obj);
        }
    }
    tracked_objects_ = remaining;

    return tracked_objects_;
}