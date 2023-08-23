/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file detection_app_c_api.cpp
 * @brief This example demonstrates running inference with virtual streams using the Hailort's C API on yolov5m
 **/

#include "multi_stream_app.hpp"

 
bool Display;
int numofStreams;


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
    RuntimeMeasure* rm = RuntimeMeasure::getInstance();
    int num_of_frames = 0;
    auto status = HAILO_SUCCESS;
    std::sort(features.begin(), features.end(), &FeatureData::sort_tensors_by_size);

    while (true)
    {
        // Gather the features into HailoTensors in a HailoROIPtr
        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        for (uint j = 0; j < features.size(); j++)
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t *>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
            
        
        
        
        rm->startTimer(TIMER_2);
        

        post_process_fun(roi);
        
        num_of_frames++;
        double time = rm->endTimer(TIMER_1);
        // std::cout << "hailo+postprocess FPS: " << (num_of_frames/time)*1000.0 << std::endl;
        double time1 = rm->endTimer(TIMER_2);
        // std::cout << "postprocess milliseconds: " << time1 << std::endl;       
        
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
    //start time measurement
    RuntimeMeasure* rm = RuntimeMeasure::getInstance();
    rm->startTimer(TIMER_1);
    // int numStreams = captures.size();
    std::vector<int> numStreams(captures.size());
    std::iota(numStreams.begin(), numStreams.end(), 0);
    while (true) {
        cv::Mat org_frame;
        bool endReached = false;
        for (int &i : numStreams) {
            captures[i] >> org_frame;
            
            //find if stream ended
            if (org_frame.empty()) {
                numStreams.erase(std::remove(numStreams.begin(), numStreams.end(), i), numStreams.end());
                if(numStreams.size() == 0){
                    endReached = true;
                    std::printf("finished reading all streams\n");
                }
                break;
            }

            cv::Mat resized_image;
            image_resize(resized_image,org_frame, IMAGE_WIDTH, IMAGE_HEIGHT);
            

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
        if(i%3 == 0)
            captures.push_back(cv::VideoCapture( VideoPath0));  
            // captures.push_back(cv::VideoCapture("v4l2src device=/dev/video0 io-mode=mmap ! video/x-raw,format=NV12,width=1920,height=1080, framerate=60/1 ! appsink", cv::CAP_GSTREAMER));  
        else if(i%3 == 1)
            captures.push_back(cv::VideoCapture( VideoPath1));
        else
            captures.push_back(cv::VideoCapture( VideoPath2));  
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
    hailo_vdevice vdevice = NULL;
    hailo_hef hef = NULL;
    hailo_configure_params_t config_params = {0};
    hailo_configured_network_group network_group = NULL;
    size_t network_group_size = 1;
    hailo_input_vstream_params_by_name_t input_vstream_params[INPUT_COUNT] = {0};
    hailo_output_vstream_params_by_name_t output_vstream_params[OUTPUT_COUNT] = {0};
    size_t input_vstreams_size = INPUT_COUNT;
    size_t output_vstreams_size = OUTPUT_COUNT;
    hailo_input_vstream input_vstreams[INPUT_COUNT] = {NULL};
    hailo_output_vstream output_vstreams[OUTPUT_COUNT] = {NULL};
    bool quantized = true;

    status = hailo_create_vdevice(NULL, &vdevice);
    REQUIRE_SUCCESS(status, l_exit, "Failed to create vdevice");

    status = hailo_create_hef_file(&hef, HEF_FILE);
    REQUIRE_SUCCESS(status, l_release_vdevice, "Failed reading hef file");

    status = hailo_init_configure_params_by_vdevice(hef, vdevice, &config_params);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed initializing configure parameters");

    status = hailo_configure_vdevice(vdevice, hef, &config_params, &network_group, &network_group_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed configure vdevice from hef");
    REQUIRE_ACTION(network_group_size == 1, status = HAILO_INVALID_ARGUMENT, l_release_hef, 
        "Invalid network group size");


    // Set input format type to auto, and mark the data as quantized - libhailort will not scale the data before writing to the HW
    quantized = true;
    status = hailo_make_input_vstream_params(network_group, quantized, HAILO_FORMAT_TYPE_AUTO,
        input_vstream_params, &input_vstreams_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed making input virtual stream params");

    /* The input format order in the example HEF is NHWC in the user-side (may be seen using 'hailortcli parse-hef <HEF_PATH>).
       Here we override the user-side format order to be NCHW */
    for (size_t i = 0 ; i < input_vstreams_size; i++) {
        input_vstream_params[i].params.user_buffer_format.order = HAILO_FORMAT_ORDER_AUTO;
    }

    // Set output format type to float32, and mark the data as not quantized - libhailort will de-quantize the data after reading from the HW
    // Note: this process might affect the overall performance
    quantized = true;
    status = hailo_make_output_vstream_params(network_group, quantized, HAILO_FORMAT_TYPE_AUTO,
        output_vstream_params, &output_vstreams_size);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed making output virtual stream params");

    REQUIRE_ACTION((input_vstreams_size <= INPUT_COUNT || output_vstreams_size <= OUTPUT_COUNT),
        status = HAILO_INVALID_OPERATION, l_release_hef, "Trying to infer network with too many input/output virtual "
        "streams, (either change HEF or change the definition of INPUT_COUNT, OUTPUT_COUNT)\n");

    status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstreams_size, input_vstreams);
    REQUIRE_SUCCESS(status, l_release_hef, "Failed creating virtual input streams\n");

    status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstreams_size, output_vstreams);
    REQUIRE_SUCCESS(status, l_release_input_vstream, "Failed creating output virtual streams\n");

    status = run_inference_threads(input_vstreams[0], output_vstreams, output_vstreams_size);
    REQUIRE_SUCCESS(status, l_release_output_vstream, "Inference failure");

    std::printf("Inference ran successfully\n");
    status = HAILO_SUCCESS;

l_release_output_vstream:
    (void)hailo_release_output_vstreams(output_vstreams, output_vstreams_size);
l_release_input_vstream:
    (void)hailo_release_input_vstreams(input_vstreams, input_vstreams_size);
l_release_hef:
    (void) hailo_release_hef(hef);
l_release_vdevice:
    (void) hailo_release_vdevice(vdevice);
l_exit:
    return status;





}


void parse_args(int argc, char* argv[])
{
        cxxopts::Options options("MyProgram", "A brief description");
        
        options.add_options()
            ("h,help", "Print help")
            ("s,num_fo_streams", "number of streams to run ", cxxopts::value<int>()->default_value("4"))
            ("d, display", "Display output (true or false)", cxxopts::value<std::string>()->default_value("true"));
            
        
        auto result = options.parse(argc, argv);
        
        if (result.count("help")) 
        {
                std::cout << options.help() << std::endl;
                exit(0);
        }
     
        std::string displayStr = result["display"].as<std::string>();
        if (displayStr == "true") {
            Display = true;
        } else if (displayStr == "false") {
            Display = false;
        } else {
            std::cerr << "Invalid display argument: " << displayStr << ". please use -h" << std::endl;
            exit(1);
        }
        numofStreams = result["num_fo_streams"].as<int>(); 


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
    parse_args(argc, argv);
    std::printf("display: %s\n", Display ? "true" : "false");
    std::printf("number of streams: %d\n", numofStreams);

    hailo_status status;
    status = infer();
    if (HAILO_SUCCESS != status)
    {
        std::cerr << "Inference failed with status = " << status << std::endl;
        return status;
    }

    return HAILO_SUCCESS;
}

