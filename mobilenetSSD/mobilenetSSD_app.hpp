/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file detection_app_c_api.cpp
 * @brief This example demonstrates running inference with virtual streams using the Hailort's C API on yolov5m
 **/


#include "hailo/hailort.h"
#include "double_buffer.hpp"

#include "mobilenet_ssd.hpp"
#include "hailo_objects.hpp"
#include "hailo_tensors.hpp"
#include "hailomat.hpp"
#include "overlay.hpp"


#include <algorithm>
#include <future>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <chrono>
#include <cxxopts.hpp>

#define INPUT_COUNT (1)
#define OUTPUT_COUNT (6)
#define INPUT_FILES_COUNT (95)

#define HEF_FILE "..\\..\\network_hef\\ssd_mobilenet_v1.hef"
#define VideoPath "..\\..\\input_images\\detection.mp4"

#define IMAGE_WIDTH 300
#define IMAGE_HEIGHT 300
#define MAX_BOXES 50

#define SaveFrames false

#define REQUIRE_ACTION(cond, action, label, ...)     \
    do                                               \
    {                                                \
        if (!(cond))                                 \
        {                                            \
            std::cout << (__VA_ARGS__) << std::endl; \
            action;                                  \
            goto label;                              \
        }                                            \
    } while (0)

#define REQUIRE_SUCCESS(status, label, ...) REQUIRE_ACTION((HAILO_SUCCESS == (status)), , label, __VA_ARGS__)
enum SourceType
{
    images,
    video,
    camera
};
class FeatureData
{
public:
    FeatureData(uint32_t buffers_size, float32_t qp_zp, float32_t qp_scale, uint32_t width, hailo_vstream_info_t vstream_info) : m_buffers(buffers_size), m_qp_zp(qp_zp), m_qp_scale(qp_scale), m_width(width), m_vstream_info(vstream_info)
    {
    }
    static bool sort_tensors_by_size(std::shared_ptr<FeatureData> i, std::shared_ptr<FeatureData> j) { return i->m_width < j->m_width; };

    DoubleBuffer<uint8_t> m_buffers;
    float32_t m_qp_zp;
    float32_t m_qp_scale;
    uint32_t m_width;
    hailo_vstream_info_t m_vstream_info;
};
hailo_status image_resize(cv::Mat &resized_image,cv::Mat from_image, int image_width, int image_height);
hailo_status display_image(HailoRGBMat &image, HailoROIPtr roi);
hailo_status create_feature(hailo_output_vstream vstream,
                            std::shared_ptr<FeatureData> &feature);
hailo_status dump_detected_object(const HailoDetectionPtr &detection, std::ofstream &detections_file);
hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData>> &features, const size_t frames_count,
                                 std::vector<HailoRGBMat> &input_images, std::queue<cv::Mat>& frameQueue, std::mutex& queueMutex);
hailo_status write_image(HailoRGBMat &image, HailoROIPtr roi);
hailo_status write_txt_file(HailoROIPtr roi, std::string file_name);
hailo_status write_all(hailo_input_vstream input_vstream, std::vector<HailoRGBMat> &input_images, std::queue<cv::Mat>& frameQueue, std::mutex& queueMutex);
hailo_status read_all(hailo_output_vstream output_vstream, const size_t frames_count, std::shared_ptr<FeatureData> feature);
hailo_status run_inference_threads(hailo_input_vstream input_vstream, hailo_output_vstream *output_vstreams,
                                   const size_t output_vstreams_size, std::vector<HailoRGBMat> &input_images);
hailo_status infer(std::vector<HailoRGBMat> &input_images);
hailo_status get_images(std::string source_path, std::vector<HailoRGBMat> &input_images, const size_t inputs_count, int image_width, int image_height);

std::vector<std::string> get_files_in_dir(const std::string& dir_path);
bool is_video(const std::string& path);
bool is_folder_of_images(const std::string& path);
