/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file detection_app_c_api.cpp
 * @brief This example demonstrates running inference with virtual streams using the Hailort's C API on yolov5m
 **/

#include "multi_stream_app.hpp"

 
bool Display = true;


hailo_status create_feature(hailo_output_vstream vstream,
                            std::shared_ptr<FeatureData> &feature)
{
    hailo_vstream_info_t vstream_info = {};
    auto status = hailo_get_output_vstream_info(vstream, &vstream_info);
    if (HAILO_SUCCESS != status)
    {
        std::cerr << "Failed to get output vstream info with status = " << status << std::endl;
        return status;
    }


    size_t output_frame_size = 0;
    status = hailo_get_output_vstream_frame_size(vstream, &output_frame_size);
    if (HAILO_SUCCESS != status)
    {
        std::cerr << "Failed getting output virtual stream frame size with status = " << status << std::endl;
        return status;
    }

    feature = std::make_shared<FeatureData>(static_cast<uint32_t>(output_frame_size), vstream_info.quant_info.qp_zp,
                                            vstream_info.quant_info.qp_scale, vstream_info.shape.width, vstream_info);

    return HAILO_SUCCESS;
}

void post_process_fun(HailoROIPtr roi)
{
    #ifdef YOLOV5_APP
        YoloParams *init_params = init(CONFIG_FILE, "yolov5");
        yolov5(roi, init_params);
    #elif defined(POSE_EST_APP)
        centerpose_416(roi);
    #elif defined(SEMANTIC_APP)
        filter(roi);
    #elif defined(INSTANCE_SEG_APP)
        YolactParams *init_params = init("dont_care","dont_care");
        yolact800mf(roi,init_params);
    #elif defined(MOBILENETSSD_APP)
        mobilenet_ssd(roi);
    #endif 
}

hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData>> &features,
                                 std::queue<cv::Mat>& frameQueue, std::queue<int>& frameIdQueue, std::mutex& queueMutex,std::vector<std::shared_ptr<SynchronizedQueue>> frameQueues)
{
    auto status = HAILO_SUCCESS;
    std::sort(features.begin(), features.end(), &FeatureData::sort_tensors_by_size);

    while (true)
    {
        // Gather the features into HailoTensors in a HailoROIPtr
        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        for (uint j = 0; j < features.size(); j++)
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t *>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
            
        
        // auto start = std::chrono::high_resolution_clock::now();
        
        // Perform the actual postprocess
        // yolov5(roi, init_params);
        post_process_fun(roi);

        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        // std::cout << "Postprosses On Host Runtime: " << duration.count() << " milliseconds" << std::endl;
        
        for (auto &feature : features)
        {
            feature->m_buffers.release_read_buffer();
        }
        
        // Draw the results
        if(Display)
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            cv::Mat frame = frameQueue.front();
            int stream_id = frameIdQueue.front(); 
            frameQueue.pop();
            frameIdQueue.pop();
            if (frame.empty() || stream_id == -1)
                break;
            HailoRGBMat image = HailoRGBMat(frame, "out" + std::to_string(stream_id));
            display_image(image, roi, stream_id, frameQueues);
        }
        
    }
    return status;
}
hailo_status display_image(HailoRGBMat &image, HailoROIPtr roi, int stream_id, std::vector<std::shared_ptr<SynchronizedQueue>> frameQueues)
{
    std::string name = image.get_name();   
    // Draw the results
    auto draw_status = draw_all(image, roi, true);
    if (OVERLAY_STATUS_OK != draw_status)
    {
        std::cerr << "Failed drawing detections on image '" << name << "'. Got status " << draw_status << "\n";
    }
    // covert back to BGR
    cv::Mat write_mat;
    cv::cvtColor(image.get_mat(), write_mat, cv::COLOR_RGB2BGR);
    
    frameQueues[stream_id]->push(write_mat.clone());
    return HAILO_SUCCESS;
}



hailo_status write_all(hailo_input_vstream input_vstream, std::queue<cv::Mat>& frameQueue, std::queue<int>& frameIdQueue, std::mutex& queueMutex, std::vector<cv::VideoCapture>& captures)
{
    int numStreams = captures.size();
    while (true) {
        cv::Mat org_frame;
        bool endReached = false;
        for (int i = 0; i < numStreams; i++) {
            captures[i] >> org_frame;
            if (org_frame.empty()) {
                numStreams--;
                if(numStreams == 0){
                    break;
                    endReached = true;
                }
            }
            cv::Mat resized_image;
            image_resize(resized_image,org_frame, IMAGE_WIDTH, IMAGE_HEIGHT);
            
            // cv::imshow("input", resized_image);
            // cv::waitKey(1);
            hailo_status status = hailo_vstream_write_raw_buffer(input_vstream, resized_image.data, resized_image.total() * resized_image.elemSize());
            if (HAILO_SUCCESS != status)
            {
                std::cerr << "Failed writing to device data of image. Got status = " << status << std::endl;
                return status;
            } 
            std::lock_guard<std::mutex> lock(queueMutex);
            frameIdQueue.push(i);
            frameQueue.push(resized_image.clone());
            
        }
        if (endReached)
            break; 
    }
    // Release the video capture resources
    for (cv::VideoCapture& capture : captures) {
        capture.release();
    }
    
    return HAILO_SUCCESS;
}

hailo_status read_all(hailo_output_vstream output_vstream, std::shared_ptr<FeatureData> feature)
{ 
    while(true)
    {
        auto &buffer = feature->m_buffers.get_write_buffer();
        hailo_status status = hailo_vstream_read_raw_buffer(output_vstream, buffer.data(), buffer.size());
        feature->m_buffers.release_write_buffer();

        if (HAILO_SUCCESS == status){
            continue;
        }
        std::cerr << "Failed reading with status = " << status << std::endl;
        return status;
        
    }

    return HAILO_SUCCESS;
}
hailo_status run_inference_threads(hailo_input_vstream input_vstream, hailo_output_vstream *output_vstreams,
                                   const size_t output_vstreams_size)
{
    // Create features data to be used for post-processing
    std::vector<std::shared_ptr<FeatureData>> features;
    features.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++)
    {
        std::shared_ptr<FeatureData> feature(nullptr);
        auto status = create_feature(output_vstreams[i], feature);
        if (HAILO_SUCCESS != status)
        {
            std::cerr << "Failed creating feature with status = " << status << std::endl;
            return status;
        }

        features.emplace_back(feature);
    }

    // Create read threads, for each output of the network (aka out vstream) we have a thread that reads the output
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++)
    {
        output_threads.emplace_back(std::async(read_all, output_vstreams[i], features[i]));
    }
      
    
    std::vector<cv::VideoCapture> captures;
    for (int i = 0; i < numofStreams; i++) {
        if(i%2 == 0)
            captures.push_back(cv::VideoCapture( VideoPath0));  
        else
            captures.push_back(cv::VideoCapture( VideoPath1));  
    }
      

    int numStreams = captures.size();
    std::vector<std::shared_ptr<SynchronizedQueue>> frameQueues;
    for (int i = 0; i < numStreams; i++) {
        auto queue = std::make_shared<SynchronizedQueue>(i);
        frameQueues.push_back(queue);
    }

    // Create a thread for running the demux Streams and display function
    std::thread([&numStreams, &frameQueues](){
        DemuxStreams demuxStreams(numStreams, frameQueues);
        demuxStreams.readAndDisplayStreams();
        }).detach();

    //queue to send the original frame from "write_all" to "post_prossing_all" for drawing the detections
    std::queue<cv::Mat> frameQueue;
    std::queue<int> streamIdQueue;
    std::mutex queueMutex;

    // Create write thread that will write the input to the device
    auto input_thread(std::async(write_all, input_vstream, std::ref(frameQueue), std::ref(streamIdQueue), std::ref(queueMutex), std::ref(captures)));
    
    // Create post-process thread that will get features from the read threads and do post-processing
    auto pp_thread(std::async(post_processing_all, std::ref(features), std::ref(frameQueue), std::ref(streamIdQueue), std::ref(queueMutex), frameQueues));
    

    ;
    // End threads
    hailo_status out_status = HAILO_SUCCESS;
    for (size_t i = 0; i < output_threads.size(); i++)
    {
        auto status = output_threads[i].get();
        
        if (HAILO_SUCCESS != status)
        {
            out_status = status;
        }
    }
    
    auto input_status = input_thread.get();
    auto pp_status = pp_thread.get();

    if (HAILO_SUCCESS != input_status)
    {
        std::cerr << "Write thread failed with status " << input_status << std::endl;
        return input_status;
    }
    if (HAILO_SUCCESS != out_status)
    {
        std::cerr << "Read failed with status " << out_status << std::endl;
        return out_status;
    }
    if (HAILO_SUCCESS != pp_status)
    {
        std::cerr << "Post-processing failed with status " << pp_status << std::endl;
        return pp_status;
    }

    

    return HAILO_SUCCESS;
}

hailo_status infer()
{
    hailo_status status = HAILO_UNINITIALIZED;
    hailo_device device = NULL;
    hailo_hef hef = NULL;
    hailo_configure_params_t config_params = {0};
    hailo_configured_network_group network_group = NULL;
    size_t network_group_size = 1;
    hailo_input_vstream_params_by_name_t input_vstream_params[INPUT_COUNT] = {0};
    hailo_output_vstream_params_by_name_t output_vstream_params[OUTPUT_COUNT] = {0};
    size_t input_vstreams_size = INPUT_COUNT;
    size_t output_vstreams_size = OUTPUT_COUNT;
    hailo_activated_network_group activated_network_group = NULL;
    hailo_input_vstream input_vstreams[INPUT_COUNT] = {NULL};
    hailo_output_vstream output_vstreams[OUTPUT_COUNT] = {NULL};

    status = hailo_create_pcie_device(NULL, &device);
    REQUIRE_SUCCESS(status, l_exit, "Failed to create pcie_device");

    status = hailo_create_hef_file(&hef, HEF_FILE);
    REQUIRE_SUCCESS(status, l_release_device, "Failed reading hef file");

    status = hailo_init_configure_params(hef, HAILO_STREAM_INTERFACE_PCIE, &config_params);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed initializing configure parameters");

    status = hailo_configure_device(device, hef, &config_params, &network_group, &network_group_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed configure devcie from hef");
    REQUIRE_ACTION(network_group_size == 1, status = HAILO_INVALID_ARGUMENT, l_release_hef, "Invalid network group size");

    status = hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
                                             input_vstream_params, &input_vstreams_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed making input virtual stream params");

    status = hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
                                              output_vstream_params, &output_vstreams_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed making output virtual stream params");

    REQUIRE_ACTION(((input_vstreams_size == INPUT_COUNT) || (output_vstreams_size == OUTPUT_COUNT)),
                   status = HAILO_INVALID_OPERATION, l_release_hef, "Expected one input vstream and three outputs vstreams");

    status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstreams_size, input_vstreams);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed creating input virtual streams");

    status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstreams_size, output_vstreams);
    REQUIRE_SUCCESS(status, l_release_input_vstream, "Failed creating output virtual streams");

    status = hailo_activate_network_group(network_group, NULL, &activated_network_group);
    REQUIRE_SUCCESS(status, l_release_output_vstream, "Failed activating network group");

    status = run_inference_threads(input_vstreams[0], output_vstreams, output_vstreams_size);
    REQUIRE_SUCCESS(status, l_deactivate_network_group, "Inference failure");

    status = HAILO_SUCCESS;
l_deactivate_network_group:
    (void)hailo_deactivate_network_group(activated_network_group);
l_release_output_vstream:
    (void)hailo_release_output_vstreams(output_vstreams, output_vstreams_size);
l_release_input_vstream:
    (void)hailo_release_input_vstreams(input_vstreams, input_vstreams_size);
l_release_hef:
    (void)hailo_release_hef(hef);
l_release_device:
    (void)hailo_release_device(device);
l_exit:
    return status;
}



hailo_status image_resize(cv::Mat &resized_image,cv::Mat from_image, int image_width, int image_height)
{    
    cv::resize(from_image, resized_image, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0, 0, cv::INTER_AREA);
    // Validate input image match the yolov5 net
    if ((image_width != resized_image.cols) || (image_height != resized_image.rows))
    {
        std::cerr << "wrong size! Size should be " << image_width << "x" << image_height << ", received: " << from_image.cols << "x" << from_image.rows << std::endl;
        return HAILO_INVALID_ARGUMENT;
    }

    // convert mat to rgb and save cv::mat as HailoRGBMat
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    return HAILO_SUCCESS;
}


int main(int argc, char* argv[])
{   
    hailo_status status;
    std::printf("display: %s\n", Display ? "true" : "false");
    
    status = infer();
    if (HAILO_SUCCESS != status)
    {
        std::cerr << "Inference failed with status = " << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}

