/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file detection_app_c_api.cpp
 * @brief This example demonstrates running inference with virtual streams using the Hailort's C API on yolov5m
 **/

#include "detection_app.hpp"

SourceType Source = SourceType::video; 
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

hailo_status dump_detected_object(const HailoDetectionPtr &detection, std::ofstream &detections_file)
{
    if (detections_file.fail())
    {
        return HAILO_FILE_OPERATION_FAILURE;
    }

    HailoBBox bbox = detection->get_bbox();
    detections_file << "Detection object name:          " << detection->get_label() << "\n";
    detections_file << "Detection object id:            " << detection->get_class_id() << "\n";
    detections_file << "Detection object confidence:    " << detection->get_confidence() << "\n";
    detections_file << "Detection object Xmax:          " << bbox.xmax() * IMAGE_WIDTH << "\n";
    detections_file << "Detection object Xmin:          " << bbox.xmin() * IMAGE_WIDTH << "\n";
    detections_file << "Detection object Ymax:          " << bbox.ymax() * IMAGE_HEIGHT << "\n";
    detections_file << "Detection object Ymin:          " << bbox.ymin() * IMAGE_HEIGHT << "\n"
                    << std::endl;

    return HAILO_SUCCESS;
}

hailo_status post_processing_all(std::vector<std::shared_ptr<FeatureData>> &features, const size_t frames_count,
                                 std::vector<HailoRGBMat> &input_images, std::queue<cv::Mat>& frameQueue, std::mutex& queueMutex)
{
    auto status = HAILO_SUCCESS;
    
    YoloParams *init_params = init(CONFIG_FILE, "yolov5");
    std::sort(features.begin(), features.end(), &FeatureData::sort_tensors_by_size);

    for (size_t i = 0; (Source==SourceType::video) || (Source == SourceType::camera) || i < frames_count; i++)
    {
        // Gather the features into HailoTensors in a HailoROIPtr
        HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));
        for (uint j = 0; j < features.size(); j++)
            roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t *>(features[j]->m_buffers.get_read_buffer().data()), features[j]->m_vstream_info));
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform the actual postprocess
         yolov5(roi, init_params);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Postprosses On Host Runtime: " << duration.count() << " milliseconds" << std::endl;
        for (auto &feature : features)
        {
            feature->m_buffers.release_read_buffer();
        }
        // get the frame from source
        HailoRGBMat image = HailoRGBMat( cv::Mat(1,1, 0) , "dummy");
        if(Source==SourceType::video || Source == SourceType::camera)
        {
            
            std::lock_guard<std::mutex> lock(queueMutex);
            if (frameQueue.empty())
            {
                std::cout<<"frameQueue.empty()"<<std::endl;
                continue;
            }
            
            cv::Mat frame = frameQueue.front();
            frameQueue.pop();
            if (frame.empty())
                break;
            image = HailoRGBMat(frame, "out" + std::to_string(i));
        }
        else if(Source==SourceType::images)
        {
            image = input_images[i];
        }
        else
        {
            std::cout<<"Source is not defined"<<std::endl;
            break;
        }
        // Draw the results
        if(Display)
        {
            auto start = std::chrono::high_resolution_clock::now();
            display_image(image, roi);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "display Runtime:             " << duration.count() << " milliseconds" << std::endl;

        }
        if(SaveFrames)
        {
            status = write_image(image, roi);
            status = write_txt_file(roi, image.get_name());
        }
        
    }
    return status;
}
hailo_status display_image(HailoRGBMat &image, HailoROIPtr roi)
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
    
    //show the image
    cv::imshow("test",write_mat);
     
    // Wait for a keystroke in the window
    char c=(char)cv::waitKey(1);// Wait for a keystroke in the window
    if(c==27)
        return HAILO_SUCCESS;      
    return HAILO_SUCCESS;
}
hailo_status write_image(HailoRGBMat &image, HailoROIPtr roi)
{
    std::string file_name = image.get_name();
    // Draw the results
    auto draw_status = draw_all(image, roi, true);
    if (OVERLAY_STATUS_OK != draw_status)
    {
        std::cerr << "Failed drawing detections on image '" << file_name << "'. Got status " << draw_status << "\n";
    }

    // convert back to BGR
    cv::Mat write_mat;
    cv::cvtColor(image.get_mat(), write_mat, cv::COLOR_RGB2BGR);
    // write to file
    auto write_status = cv::imwrite("..\\output_images\\" + file_name + ".bmp", write_mat);
    if (true != write_status)
    {
        std::cerr << "Failed dumping image '" << file_name << std::endl;
        return HAILO_FILE_OPERATION_FAILURE;
    }
    return HAILO_SUCCESS;
}

hailo_status write_txt_file(HailoROIPtr roi, std::string file_name)
{
    std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);
    hailo_status status = HAILO_SUCCESS;

    // check if detections were found
    if (detections.size() == 0)
    {
        std::cout << "No detections were found in file '" << file_name << "'\n";
        return status;
    }

    // Prepare output files
    auto detections_file = "..\\output_images\\" + file_name + "_detections.txt";
    std::ofstream ofs(detections_file, std::ios::out);
    if (ofs.fail())
    {
        std::cerr << "Failed opening output file: '" << detections_file << "'\n";
        return HAILO_OPEN_FILE_FAILURE;
    }

    // write detections info in the text file
    for (auto &detection : detections)
    {
        if (0 == detection->get_confidence())
        {
            continue;
        }
        status = dump_detected_object(detection, ofs);
        if (HAILO_SUCCESS != status)
        {
            std::cerr << "Failed dumping detected object in output file: '" << detections_file << "'\n";
        }
    }
    return status;
}

hailo_status write_all(hailo_input_vstream input_vstream, std::vector<HailoRGBMat> &input_images, std::queue<cv::Mat>& frameQueue, std::mutex& queueMutex)
{
    if(Source==SourceType::images)
    {
        for (auto &input_image : input_images)
        {
            auto image_mat = input_image.get_mat();
            hailo_status status = hailo_vstream_write_raw_buffer(input_vstream, image_mat.data, image_mat.total() * image_mat.elemSize());
            if (HAILO_SUCCESS == status)
                continue;    
            std::cerr << "Failed writing to device data of image '" << input_image.get_name() << "'. Got status = " << status << std::endl;
            return status;
        }
    }
    else if(Source==SourceType::video || Source==SourceType::camera)
    {
        
        cv::VideoCapture cap;
        if (Source == SourceType::camera) {
            cap = cv::VideoCapture(0);
        } else {
            cap = cv::VideoCapture(VideoPath);
        }  
        
        // Check if camera opened successfully
        if(!cap.isOpened()){
            std::cout << "Error opening video stream or file" << std::endl;
            return HAILO_OPEN_FILE_FAILURE;
        }
        for (;;)
        {
            cv::Mat org_frame;
            cap >> org_frame;
            
            if (org_frame.empty())
                break;
            
            cv::Mat resized_image;
            image_resize(resized_image,org_frame, IMAGE_WIDTH, IMAGE_HEIGHT);
            
            
            hailo_status status = hailo_vstream_write_raw_buffer(input_vstream, resized_image.data, resized_image.total() * resized_image.elemSize());
            if (HAILO_SUCCESS != status)
            {
                std::cerr << "Failed writing to device data of image. Got status = " << status << std::endl;
                return status;
            } 
            std::lock_guard<std::mutex> lock(queueMutex);
            frameQueue.push(resized_image.clone());
        }
    }
    return HAILO_SUCCESS;
}

hailo_status read_all(hailo_output_vstream output_vstream, const size_t frames_count, std::shared_ptr<FeatureData> feature)
{

    size_t i = 0;
    while((Source==SourceType::video) || Source == SourceType::camera || i < frames_count  )
    {
        auto &buffer = feature->m_buffers.get_write_buffer();
        hailo_status status = hailo_vstream_read_raw_buffer(output_vstream, buffer.data(), buffer.size());
        feature->m_buffers.release_write_buffer();

        if (HAILO_SUCCESS == status){
            i++;
            continue;
        }
        std::cerr << "Failed reading with status = " << status << std::endl;
        return status;
        
    }

    return HAILO_SUCCESS;
}
hailo_status run_inference_threads(hailo_input_vstream input_vstream, hailo_output_vstream *output_vstreams,
                                   const size_t output_vstreams_size, std::vector<HailoRGBMat> &input_images)
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

    // Create read threads
    const size_t frames_count = input_images.size();
    std::vector<std::future<hailo_status>> output_threads;
    output_threads.reserve(output_vstreams_size);
    for (size_t i = 0; i < output_vstreams_size; i++)
    {
        output_threads.emplace_back(std::async(read_all, output_vstreams[i], frames_count, features[i]));
    }
      
    //queue to send the original frame from "write_all" to "post_prossing_all" for display
    std::queue<cv::Mat> frameQueue;
    std::mutex queueMutex;
   
    // Create write thread
    auto input_thread(std::async(write_all, input_vstream, std::ref(input_images), std::ref(frameQueue), std::ref(queueMutex)));
    
    // Create post-process thread
   
    auto pp_thread(std::async(post_processing_all, std::ref(features), frames_count, std::ref(input_images), std::ref(frameQueue), std::ref(queueMutex)));
    

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

hailo_status infer(std::vector<HailoRGBMat> &input_images)
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

    status = run_inference_threads(input_vstreams[0], output_vstreams, output_vstreams_size, input_images);
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


hailo_status get_images(std::string source_path, std::vector<HailoRGBMat> &input_images, const size_t inputs_count, int image_width, int image_height)
{
    for (uint32_t i = 0; i < inputs_count; i++)
    {
        std::string file_name = "out" + std::to_string(i+1);
        std::string file_path = source_path + file_name + ".png";
        cv::Mat bgr_mat = cv::imread(file_path);
        if (bgr_mat.empty())
        {
            std::cerr << "Failed reading file: " << file_path << std::endl;
            return HAILO_OPEN_FILE_FAILURE;
        }
        cv::Mat resized_image;
        image_resize(resized_image,bgr_mat, image_width, image_height);
        HailoRGBMat image = HailoRGBMat(resized_image, file_name);
        input_images.emplace_back(image);
    }
    return HAILO_SUCCESS;
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
std::vector<std::string> parse_args(int argc, char* argv[])
{
    cxxopts::Options options("MyProgram", "A brief description");
    
    options.add_options()("h,help", "Print help")(
      "f,hef", "Path to .hef file", cxxopts::value<std::string>()
      ->default_value(HEF_FILE))(
      "s,source", "Path to a video file or directory with images ", cxxopts::value<std::string>()
      ->default_value(VideoPath))(
      "d, display", "Display output (true or false)", cxxopts::value<bool>()
      ->default_value("true"))(
      "c,camera", "Use camera as the source");
      
    
    auto result = options.parse(argc, argv);
    
    if (result.count("help")) 
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
   
    
    if (result.count("camera") > 0 && result.count("source") > 0)
    {
        std::cerr << "Both camera and source options cannot be specified together" << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
    // if (!result.count("file") || !result.count("source") || !result.count("hef")) {
    //     std::cerr << "Missing required arguments" << std::endl;
    //     std::cerr << options.help() << std::endl;
    //     return 1;
    // }
    
    Display = result["display"].as<bool>();
    std::string hef_path = result["hef"].as<std::string>(); 
    std::string source_path = result["source"].as<std::string>();
    if(is_folder_of_images(source_path))
    {
        Source = SourceType::images;
    }
    else if (result.count("camera"))
    {
        Source = SourceType::camera;
    }
    else if (is_video(source_path))
    {
        Source = SourceType::video;
    }
    else
    {
        std::cerr << "Invalid source type" << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
    std::vector<std::string> args;
    args.push_back(hef_path);
    args.push_back(source_path);
    return args;
}
bool is_video(const std::string& path) {
    std::filesystem::path p(path);
    if (!std::filesystem::exists(p)) {
        return false;
    }
    if (!std::filesystem::is_regular_file(p)) {
        return false;
    }
    if (!p.has_extension()) {
        return false;
    }
    if (p.extension() != ".mp4" && p.extension() != ".avi") {
        return false;
    }
    return true;
}
bool is_folder_of_images(const std::string& path) {
    std::filesystem::path p(path);
    if (!std::filesystem::exists(p)) {
        return false;
    }
    if (!std::filesystem::is_directory(p)) {
        return false;
    }
    for (const auto& entry : std::filesystem::directory_iterator(p)) {
        if (!std::filesystem::is_regular_file(entry)) {
            return false;
        }
        if (!entry.path().has_extension()) {
            return false;
        }
        if (entry.path().extension() != ".png" && entry.path().extension() != ".jpg" && entry.path().extension() != ".png" ) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{   
    hailo_status status;
    std::vector<std::string> args = parse_args(argc, argv);
    std::string hef_path = args[0];
    std::string source_path = args[1];
    std::printf("hef_path: %s\n", hef_path.c_str());
    std::printf("source_path: %s\n", source_path.c_str());
    std::printf("source type: %s\n", Source==SourceType::images ? "images" : (Source==SourceType::video ? "video" : "camera"));
    std::printf("display: %s\n", Display ? "true" : "false");
    if(Source==SourceType::images)
    {
        std::vector<HailoRGBMat> input_images;
        input_images.reserve(INPUT_FILES_COUNT);
        auto status = get_images(source_path ,input_images, INPUT_FILES_COUNT, IMAGE_WIDTH, IMAGE_HEIGHT);
        if (HAILO_SUCCESS != status)
        {
            std::cerr << "get_images() failed to with status = " << status << std::endl;
            return status;
        }
        
        status = infer(input_images);
        if (HAILO_SUCCESS != status)
        {
            std::cerr << "Inference failed with status = " << status << std::endl;
            return status;
        }
    }
    else if(Source==SourceType::video || Source==SourceType::camera)
    {   
        std::vector<HailoRGBMat> input_images;
        
        status = infer(input_images);
        if (HAILO_SUCCESS != status)
        {
            std::cerr << "Inference failed with status = " << status << std::endl;
            return status;
        }
    }
    else
    {
        std::cerr << "Not Valid source Option" << std::endl;
    }
    return HAILO_SUCCESS;
}

