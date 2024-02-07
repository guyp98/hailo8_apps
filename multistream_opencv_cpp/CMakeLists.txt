cmake_minimum_required(VERSION 3.20.0)
project(MultiCameraDisplay)

set(CMAKE_CXX_STANDARD 20)
if(MSVC)
  message("Set optimization flags for Visual Studio")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox /fp:fast /Qfast_transcendentals")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /fp:fast /Qfast_transcendentals")
  set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
  add_definitions(/DNOMINMAX
			      /D_HAILO_EXPORTING
                  /W0
                  /DWIN32_LEAN_AND_MEAN
    )
elseif(UNIX)
  # Set optimization flags for non-Visual Studio compilers
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -finline-limit=1000000 -ffp-contract=fast -Wall -pthread")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -finline-limit=1000000 -ffp-contract=fast -Wall -pthread")
  add_definitions(/DNOMINMAX
				  /D_HAILO_EXPORTING
    )
    option(ARK "arm or x86 unix")
  if(ARC STREQUAL "x86")
    set(CMAKE_C_FLAGS "-mavx2 -mfma")
    set(CMAKE_CXX_FLAGS "-mavx2 -mfma")
  endif()
else()
    message(FATAL_ERROR "Unsupported compiler")
endif()



# include directories
find_package(OpenCV REQUIRED)
find_package(HailoRT REQUIRED)

include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)

ExternalProject_Add(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target install
)

# Build xtensor
ExternalProject_Add(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -Dxtl_DIR=${CMAKE_BINARY_DIR}/external/share/cmake/xtl
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target install
)

# Set the include and link directories for external projects
include_directories(
    ${EXTERNAL_INSTALL_LOCATION}/include
    #${EXTERNAL_INSTALL_LOCATION}/lib
)



# set the source files
set(utils_path ${CMAKE_CURRENT_SOURCE_DIR}/utils/)
set(utility_source_files
    ${utils_path}MultiStreamDisplay.cpp
    ${utils_path}SynchronizedQueue.cpp
    ${utils_path}DemuxStreams.cpp
    ${utils_path}RuntimeMeasure.cpp
    ${utils_path}PrintLock.cpp
    )


set(relative_workspace ../dependencies/)
set(overlay_file ${relative_workspace}/core/hailo/plugins/overlay/overlay.cpp)
set(post_detection ${relative_workspace}/core/hailo/libs/postprocesses/detection/yolo_postprocess.cpp)
set(post_detection2 ${relative_workspace}/core/hailo/libs/postprocesses/detection/yolo_output.cpp)
set(post_pose_est ${relative_workspace}/core/hailo/libs/postprocesses/pose_estimation/centerpose.cpp)
set(post_semantic_seg ${relative_workspace}/core/hailo/libs/postprocesses/semantic_segmentation/semantic_segmentation.cpp)
set(post_instance_seg_yolact ${relative_workspace}/core/hailo/libs/postprocesses/instance_segmentation/yolact.cpp)
set(post_instance_seg_yolov5 ${relative_workspace}/core/hailo/libs/postprocesses/instance_segmentation/yolov5_seg/yolov5seg.cpp)
set(post_mobilenet ${relative_workspace}/core/hailo/libs/postprocesses/detection/mobilenet_ssd.cpp)
set(common_tensors_file ${relative_workspace}/core/hailo/libs/postprocesses/common/tensors.cpp)


# set the include directories
include_directories(
    utils_path
    SYSTEM ${relative_workspace}/core/open_source/rapidjson
    ${relative_workspace}/core/hailo/general
    ${relative_workspace}/core/hailo/libs/postprocesses
    ${relative_workspace}/core/hailo/plugins/common
    ${relative_workspace}/core/hailo/plugins
    ${relative_workspace}/core/hailo/plugins/overlay
    ${relative_workspace}/core/open_source/cxxopts
    )

    
# select the app to build
set(apps_base_conf ./build_cmake_comfig/)
option(MACRO_SET "Select the app to build" "YOLOV5_APP or POSE_EST_APP or SEMANTIC_APP or INSTANCE_SEG_APP")
if(MACRO_SET STREQUAL "YOLOV5_APP")
    set(hailo_utils 
        ${overlay_file}
        ${post_detection}
        ${post_detection2}
        ${common_tensors_file}
        )
    include_directories(
        ${relative_workspace}/core/hailo/libs/postprocesses/detection
    )
    include(${apps_base_conf}CMakeLists_YOLOV5.txt)
elseif(MACRO_SET STREQUAL "POSE_EST_APP")
    set(hailo_utils 
        ${overlay_file}
        ${post_pose_est}
        ${common_tensors_file}
        )
    include_directories(
        ${relative_workspace}/core/hailo/libs/postprocesses/pose_estimation
    )
    include(${apps_base_conf}CMakeLists_POSE_EST.txt)
elseif(MACRO_SET STREQUAL "SEMANTIC_APP")
    set(hailo_utils 
        ${overlay_file}
        ${post_semantic_seg}
        ${common_tensors_file}
        )
    include_directories(
        ${relative_workspace}/core/hailo/libs/postprocesses/semantic_segmentation
    )
    include(${apps_base_conf}CMakeLists_SEMANTIC.txt)
elseif(MACRO_SET STREQUAL "INSTANCE_SEG_APP_YOLACT")
    set(hailo_utils 
        ${overlay_file}
        ${post_instance_seg_yolact}
        ${common_tensors_file}
        )
    include_directories(
        ${relative_workspace}/core/hailo/libs/postprocesses/instance_segmentation
    )
    include(${apps_base_conf}CMakeLists_INSTANCE_SEG_YOLACT.txt)
elseif(MACRO_SET STREQUAL "INSTANCE_SEG_APP_YOLOV5")
    set(hailo_utils 
        ${overlay_file}
        ${post_instance_seg_yolov5}
        ${common_tensors_file}
        )
    include_directories(
        ${relative_workspace}/core/hailo/libs/postprocesses/instance_segmentation/yolov5_seg
    )
    include(${apps_base_conf}CMakeLists_INSTANCE_SEG_YOLOV5.txt)
elseif(MACRO_SET STREQUAL "MOBILENETSSD_APP")
    set(hailo_utils 
        ${overlay_file}
        ${post_mobilenet}
        ${common_tensors_file}
        )
    include_directories(
        ${relative_workspace}/core/hailo/libs/postprocesses/detection
    )
    include(${apps_base_conf}CMakeLists_MOBILENETSSD.txt)
else()
    message(FATAL_ERROR "No macro set")
Endif()   

# download the model
include(${apps_base_conf}DownloadFile.cmake)

# build the app
add_executable(multi_stream_app 
            multi_stream_app.cpp
            hailo_hw_c.hpp
            ${utility_source_files}
            ${hailo_utils}
            )

target_compile_definitions(multi_stream_app PUBLIC ${MACRO_SET})

target_link_libraries(multi_stream_app PUBLIC HailoRT::libhailort ${OpenCV_LIBS} )
