/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <vector>
#include <cmath>
#include "vms_croppers.hpp"

#define PERSON_LABEL "person"
#define FACE_LABEL "face"
#define FACE_ATTRIBUTES_CROP_SCALE_FACTOR (1.58f)
#define FACE_ATTRIBUTES_CROP_HIGHT_OFFSET_FACTOR (0.10f)
#define TRACK_UPDATE 60

std::map<int, int> track_counter;

/**
* @brief Get the tracking Hailo Unique Id object from a Hailo Detection.
* 
* @param detection HailoDetectionPtr
* @return HailoUniqueIdPtr pointer to the Hailo Unique Id object
*/
HailoUniqueIDPtr get_tracking_id(HailoDetectionPtr detection)
{
    for (auto obj : detection->get_objects_typed(HAILO_UNIQUE_ID))
    {
        HailoUniqueIDPtr id = std::dynamic_pointer_cast<HailoUniqueID>(obj);
        if (id->get_mode() == TRACKING_ID)
        {
            return id;
        }
    }
    return nullptr;
}

/**
* @brief Returns a boolean box is invalid cause it has nan value.
* 
* @param box HailoBBox
* @return boolean indicating if box has nan value.
*/
bool box_contains_nan(HailoBBox box)
{
    return (std::isnan(box.xmin()) && std::isnan(box.ymin()) && std::isnan(box.width()) && std::isnan(box.height()));
}

/**
* @brief Returns a boolean indicating if traker update is required for a given detection.
*       It is determined by the number of frames since the last update.
*       How many frames to wait for an update are defined in TRACK_UPDATE.
* 
* @param detection HailoDetectionPtr
* @param use_track_update boolean can override the default behaviour, false will always require an update
* @return boolean indicating if traker update is required.
*/
bool track_update(HailoDetectionPtr detection, bool use_track_update)
{
    auto tracking_obj = get_tracking_id(detection);
    if (tracking_obj && use_track_update)
    {
        int tracking_id = tracking_obj->get_id();
        auto counter = track_counter.find(tracking_id);
        if (counter == track_counter.end())
        {
            // Emplace new element to the track_counter map. track update required.
            track_counter.emplace(tracking_id, 0);
            return true;
        }
        else if (counter->second >= TRACK_UPDATE)
        {
            // Counter passed the TRACK_UPDATE limit - set existing track to 0. track update required.
            track_counter[tracking_id] = 0;
            return true;
        }
        else if (counter->second < TRACK_UPDATE)
        {
            // Counter is still below TRACK_UPDATE_LIMIT - increasing the exising value. track update should be skipped. 
            track_counter[tracking_id] += 1;
        }

        return false;
    }

    return true;
}

/**
 * @brief Returns a vector of Person detections to crop and resize.
 *
 * @param image The original picture (cv::Mat).
 * @param roi The main ROI of this picture.
 * @return std::vector<HailoROIPtr> vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> person_crop(cv::Mat image, HailoROIPtr roi, bool use_track_update=false)
{
    std::vector<HailoROIPtr> crop_rois;
    // Get all detections.
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);
    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        // Modify only detections with "person" label.
        if (std::string(PERSON_LABEL) == detection->get_label())
        {
            if (track_update(detection, use_track_update))
                crop_rois.emplace_back(detection);
        }
    }
    return crop_rois;
}

/**
 * @brief Returns an adjusted HailoBBox acordding to 3ddfa cropping algorithm.
 *
 * @param image The original picture (cv::Mat).
 * @param roi The ROI to modify
 * @return HailoBBox Adjusted HailoBBox to crop.
 * @note Original algorithm at https://github.com/cleardusk/3DDFA_V2/blob/9fdbea1eb97f762221f71f5c76f08f52296c6704/utils/functions.py#L85
 */
HailoBBox algorithm_face_crop(cv::Mat &image, const HailoBBox &roi, float size_scale = 1.0, float height_offset = 0.0)
{
    // Algorithm
    float old_size = (roi.width() * image.cols + roi.height() * image.rows) / 2;
    float center_x = (2 * roi.xmin() + roi.width()) / 2;
    float center_y = (2 * roi.ymin() + roi.height()) / 2 - (old_size / image.rows) * height_offset;
    float size = old_size * size_scale;
    // Determine the new width and height, height should be a little bigger.
    float h_size = size / image.rows;
    float w_size = size / image.cols;
    // Determine the top left corner of the crop.
    float xmin = CLAMP((center_x - w_size / 2), 0, 1);
    float ymin = CLAMP((center_y - h_size / 2), 0, 1);
    return HailoBBox(xmin, ymin, CLAMP(w_size, 0, 1 - xmin), CLAMP(h_size, 0, 1 - ymin));
    // return HailoBBox(roi.xmin(), roi.ymin(), roi.width(), roi.height());
}

HailoDetectionPtr clone_detection_object(HailoDetectionPtr detection)
{
    HailoDetectionPtr new_roi = std::make_shared<HailoDetection>(detection->get_bbox(), detection->get_label(), detection->get_confidence());

    for (auto object : detection->get_objects())
    {
        HailoObjectPtr new_object;
        switch (object->get_type())
        {
            case HAILO_LANDMARKS:
            {
                auto landmarks = std::dynamic_pointer_cast<HailoLandmarks>(object);
                new_object = landmarks->clone();
                break;
            }
            case HAILO_CLASSIFICATION:
            {
                auto classification = std::dynamic_pointer_cast<HailoClassification>(object);
                new_object = classification->clone();
                break;
            }
            case HAILO_MATRIX:
            {
                auto matrix = std::dynamic_pointer_cast<HailoMatrix>(object);
                new_object = matrix->clone();
                break;
            }
            case HAILO_DETECTION:
            {
                auto detection = std::dynamic_pointer_cast<HailoDetection>(object);
                new_object = detection->clone();
                break;
            }
            case HAILO_UNIQUE_ID:
            {
                auto unique_id = std::dynamic_pointer_cast<HailoUniqueID>(object);
                new_object = unique_id->clone();
                break;
            }
            default:
                break;
        }

        new_roi->add_object(new_object);
    }

    return new_roi;
}

/**
 * @brief Returns a vector of face detections to crop and resize.
 *
 * @param image The original picture (cv::Mat).
 * @param roi The main ROI of this picture.
 * @param track_update update track every X frames.
 * @return std::vector<HailoROIPtr> vector of ROI's to crop and resize.
 */
std::vector<HailoROIPtr> face_crop(cv::Mat image, HailoROIPtr roi, bool use_track_update=false)
{
    std::vector<HailoROIPtr> crop_rois;
    // Get all detections.
    std::vector<HailoDetectionPtr> detections_ptrs = hailo_common::get_hailo_detections(roi);
    for (HailoDetectionPtr &detection : detections_ptrs)
    {
        // Modify only detections with "face" label.
        if (std::string(FACE_LABEL) == detection->get_label() && !box_contains_nan(detection->get_bbox()))
        {
            if (track_update(detection, use_track_update))
            {
                // Modifies a rectengle according to a cropping algorithm only on faces
                auto new_bbox = algorithm_face_crop(image, detection->get_bbox(), FACE_ATTRIBUTES_CROP_SCALE_FACTOR, FACE_ATTRIBUTES_CROP_HIGHT_OFFSET_FACTOR);

                HailoDetectionPtr new_roi = clone_detection_object(detection);
                hailo_common::fixate_landmarks_with_bbox(new_roi, new_bbox);

                new_roi->set_bbox(new_bbox);
                crop_rois.emplace_back(new_roi);
            }
        }
    }
    return crop_rois;
}

std::vector<HailoROIPtr> face_recognition(cv::Mat image, HailoROIPtr roi)
{
    return face_crop(image, roi, false);
}

std::vector<HailoROIPtr> face_attributes(cv::Mat image, HailoROIPtr roi)
{
    return face_crop(image, roi, true);
}

std::vector<HailoROIPtr> person_attributes(cv::Mat image, HailoROIPtr roi)
{
    return person_crop(image, roi, true);
}