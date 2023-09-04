# Desctiption
Run multipule video or camera streams on one hailo chip on ubuntu or windows. 
The app takes multipule streams and muxes them to the hailo chip. 
The output stream from the chip then demuxed and sent for display. 

## Build Environment for Ubuntu
1. Install CMake version 3.23.0 and up.
2. Install hailort (minimum version 4.14) from hailo site.
3. Install opencv for c++.
2. Install hailort from hailo site (https://www.hailo.ai/).
3. Install opencv for c++ (https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html).
5. Clone the project.
6. run the following commands:
   ```
   mkdir build
   cd build
   cmake -DMACRO_SET={network to run} ..
   cmake --build . --config Release
   .\multi_stream_app
  * {network to run} - YOLOV5_APP or POSE_EST_APP or SEMANTIC_APP or INSTANCE_SEG_APP or MOBILENETSSD_APP .
   for example "cmake -DMACRO_SET=YOLOV5_APP .." ```
  * also you can add an optimization for x86 architecture like this "cmake -DMACRO_SET={network to run} -DARK=x86 .. "
    
## Build Environment Windows
1. Install CMake version 3.23.0 and up.
2. Install the latest `hailort.msi` from the Hailo website.
3. Install the latest `C++ opencv.msi` for Windows.
4. Make sure that OpenCV and Hailo Runtime are added to the Windows PATH.
5. Clone the project.
6. run the following commands (in "x64 Native Tool Command Prompt for VS" cmd):
    ```
    mkdir build
    cd build
    cmake -DMACRO_SET={network to run} ..
    cmake --build . --config Release
    .\Release\multi_stream_app.exe
    ```
* {network to run} - YOLOV5_APP or POSE_EST_APP or SEMANTIC_APP or INSTANCE_SEG_APP or MOBILENETSSD_APP .
   for example "cmake -DMACRO_SET=YOLOV5_APP .."
## Common Pitfalls with Windows
* If the project path is too long, you may encounter an error. It is recommended to keep your path short.
* Running the project from a path with spaces in its name can sometimes cause unexpected errors.
