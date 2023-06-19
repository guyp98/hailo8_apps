cmake_minimum_required(VERSION 3.22.0)
project(semantic_app
        LANGUAGES C CXX
        )

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
if(MSVC)
  message("Set optimization flags for Visual Studio")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Ox /arch:AVX2 /fp:fast /Qfast_transcendentals /Qpar /Qpar-report:2 ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox /arch:AVX2 /fp:fast /Qfast_transcendentals /Qpar /Qpar-report:2  ")
  set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
  add_definitions(/DNOMINMAX
				        /D_HAILO_EXPORTING
                /W0
                /DWIN32_LEAN_AND_MEAN
                 )
else()
  # Set optimization flags for non-Visual Studio compilers
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx2 -mfma -finline-limit=1000000 -ffp-contract=fast -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -finline-limit=1000000 -ffp-contract=fast -Wall")
  add_definitions(/DNOMINMAX
				/D_HAILO_EXPORTING
        )
endif()


# Set the build type to release
set(CMAKE_BUILD_TYPE "Release")

# Define the source files
set(relative_workspace ../dependencies)
set(overlay_file ${relative_workspace}/core/hailo/plugins/overlay/overlay.cpp)
set(post_file ${relative_workspace}/core/hailo/libs/postprocesses/semantic_segmentation/semantic_segmentation.cpp)
set(common_tensors_file ${relative_workspace}/core/hailo/libs/postprocesses/common/tensors.cpp)

# Include directories
include_directories(
  SYSTEM core/open_source/rapidjson
  ${relative_workspace}/core/hailo/general
  ${relative_workspace}/core/hailo/libs/postprocesses
  ${relative_workspace}/core/hailo/libs/postprocesses/semantic_segmentation
  ${relative_workspace}/core/hailo/plugins/common
  ${relative_workspace}/core/hailo/plugins
  ${relative_workspace}/core/hailo/plugins/overlay
  ${relative_workspace}/core/open_source/cxxopts
)

# Include ExternalProject module
include(ExternalProject)

# Set the installation location for external projects
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)


# Build xtl
ExternalProject_Add(xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}/xtl
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target install
)

# Build xtensor
ExternalProject_Add(xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}/xtensor -Dxtl_DIR=${CMAKE_BINARY_DIR}/build/external/xtl/share/cmake/xtl
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target install
)


# Set the include and link directories for external projects
include_directories(
    ${EXTERNAL_INSTALL_LOCATION}/xtl/include
    ${EXTERNAL_INSTALL_LOCATION}/xtensor/include
)
link_directories(
    ${EXTERNAL_INSTALL_LOCATION}/xtl/lib
    ${EXTERNAL_INSTALL_LOCATION}/xtensor/lib
)

# Find the required packages
find_package(OpenCV REQUIRED)
# find_package(xtensor REQUIRED) 
# find_package(xtl REQUIRED)
find_package(HailoRT REQUIRED)



# Add the executable
add_executable(semantic_app 
    semantic_app.cpp
    ${overlay_file}
    ${post_file}
    ${common_tensors_file}
    )

# Run get_resourced.bat after the build is complete
# add_custom_command(
#     TARGET semantic_app
#     POST_BUILD
#     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/get_resources.bat
# )

target_link_libraries(semantic_app PUBLIC HailoRT::libhailort)
target_link_libraries(semantic_app PRIVATE ${OpenCV_LIBS})
# target_link_libraries(semantic_app PUBLIC xtensor -w xtensor::optimize  xtl)

