# Download the hef files
set(hef_names
    "centerpose_regnetx_1.6gf_fpn.hef"
    "centerpose_repvgg_a0.hef"
    "fcn8_resnet_v1_18.hef"
    "ssd_mobilenet_v1.hef"
    "yolact_regnetx_800mf.hef"
    "yolov5m_wo_spp_60p.hef"
)
set(hef_urls
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/multistream+app+(guy)/centerpose_regnetx_1.6gf_fpn.hef"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/multistream+app+(guy)/centerpose_repvgg_a0.hef"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/multistream+app+(guy)/fcn8_resnet_v1_18.hef"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/multistream+app+(guy)/ssd_mobilenet_v1.hef"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/multistream+app+(guy)/yolact_regnetx_800mf.hef"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/multistream+app+(guy)/yolov5m_wo_spp_60p.hef"
)

# Define the directory to download the files to
set(download_dir "../../network_hef")

# Create the directory if it doesn't exist
file(MAKE_DIRECTORY ${download_dir})



list(LENGTH hef_names list_length)
math(EXPR list_length "${list_length} - 1")

# Download each hef file from the lists
foreach(i RANGE 0 ${list_length})
    
    list(GET hef_names ${i} filename)
    list(GET hef_urls ${i} url)
    
    # Check if the file already exists in the download directory
    if(NOT EXISTS "${download_dir}/${filename}")
        message(STATUS "Downloading ${filename} from ${url}")
        file(DOWNLOAD ${url} "${download_dir}/${filename}")
    endif()
endforeach()





#Download video files
set(video_names
    "car_drive.mp4"
    "detection.mp4"
    "river_tiber1280x1024.m4v"
    )
set(videos_urls 
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/multistream+app+(guy)/car_drive.mp4"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/multistream+app+(guy)/detection.mp4"
    "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/multistream+app+(guy)/river_tiber1280x1024.m4v"
    )

# Define the directory to download the files to
set(download_dir "../../input_images")

# Create the directory if it doesn't exist
file(MAKE_DIRECTORY ${download_dir})

list(LENGTH video_names list_length)
math(EXPR list_length "${list_length} - 1")


# Download each hef file from the lists
foreach(i RANGE 0 ${list_length})
    
    list(GET video_names ${i} filename)
    list(GET videos_urls ${i} url)

    # Check if the file already exists in the download directory
    if(NOT EXISTS "${download_dir}/${filename}")
        message(STATUS "Downloading ${filename} from ${url}")
        file(DOWNLOAD ${url} "${download_dir}/${filename}")
    endif()
endforeach()



