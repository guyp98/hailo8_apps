/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file detection_app_c_api.cpp
 * @brief This example demonstrates running inference with virtual streams using the Hailort's C API on yolov5m
 **/

#include "hailo/hailort.h"
#include "utils/double_buffer.hpp"
#include "yolo_postprocess.hpp"
#include "hailo_objects.hpp"
#include "hailo_tensors.hpp"
#include "hailomat.hpp"
#include "overlay.hpp"
#include "utils/SynchronizedQueue.hpp"
#include "utils/DemuxStreams.hpp"

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
#include <thread>



#define INPUT_COUNT (1)
#define OUTPUT_COUNT (3)
#define INPUT_FILES_COUNT (10)
#define HEF_FILE "../../network_hef/yolov5m_wo_spp_60p.hef"
#define VideoPath "../../input_images/detection.mp4"

#define CONFIG_FILE "../yolov5.json"
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 640
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
hailo_status display_image(HailoRGBMat &image, HailoROIPtr roi, int stream_id, std::vector<std::shared_ptr<SynchronizedQueue>> frameQueues);
hailo_status create_feature(hailo_output_vstream vstream,
                            std::shared_ptr<FeatureData> &feature);
hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData>> &features, 
                                 std::queue<cv::Mat>& frameQueue, std::queue<int>& frameIdQueue, std::mutex& queueMutex, int numStreams);

hailo_status write_all(hailo_input_vstream input_vstream, std::queue<cv::Mat>& frameQueue, std::queue<int>& frameIdQueue, std::mutex& queueMutex, std::vector<cv::VideoCapture>& captures);
hailo_status read_all(hailo_output_vstream output_vstream,  std::shared_ptr<FeatureData> feature);
hailo_status run_inference_threads(hailo_input_vstream input_vstream, hailo_output_vstream *output_vstreams,
                                   const size_t output_vstreams_size);
hailo_status infer();


std::vector<std::string> get_files_in_dir(const std::string& dir_path);
bool is_video(const std::string& path);
bool is_folder_of_images(const std::string& path);