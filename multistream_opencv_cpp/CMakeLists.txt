cmake_minimum_required(VERSION 3.20.0)
project(MultiCameraDisplay)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

set(utils_path ${CMAKE_CURRENT_SOURCE_DIR}/utils/)
set(utility_source_files
    ${utils_path}MultiStreamDisplay.cpp
    ${utils_path}SynchronizedQueue.cpp
    ${utils_path}DemuxStreams.cpp
    )

# add_executable(MultiCameraDisplay multi_stream.cpp)
add_executable(Displaytest 
            Displaytest.cpp
            ${utility_source_files})

# target_link_libraries(MultiCameraDisplay ${OpenCV_LIBS})
target_link_libraries(Displaytest ${OpenCV_LIBS})