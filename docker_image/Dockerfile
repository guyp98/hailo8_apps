
FROM ubuntu:22.04


RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ubuntu-desktop \
    build-essential \
    git \
    cmake \
    vim \
    tmux \ 
    sudo \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module &&\
    apt-get clean


WORKDIR /app


#build hailo driver
RUN mkdir hailo_drivers
COPY hailort_4.14.0_amd64.deb /app/hailo_drivers/
RUN sudo touch /.dockerenv &&\ 
    # echo 'y\n' | dpkg -i /app/hailo_drivers/hailort_4.14.0_amd64.deb
    echo 'y\n' | dpkg -i /app/hailo_drivers/hailort_4.14.0_amd64.deb &&\
    rm -rf /app/hailo_drivers



#build opencv 

#1)Install the build tools and dependencies
RUN echo "Y\n" | apt-get install pkg-config libgtk-3-dev libavcodec-dev\
    libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev



#2)Download and build OpenCV
RUN mkdir opencv_build && cd opencv_build &&\
    git clone https://github.com/opencv/opencv.git &&\
    cd opencv &&\
    mkdir build && cd build &&\
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=ON .. &&\
    make -j$(($(nproc) * 3 / 4)) &&\
    make install
    # make install &&\
    # rm -rf /app/opencv_build


#3)Download and build muiltistream app
RUN cd /app &&\
    git clone https://github.com/guyp98/hailo_windows_apps.git &&\
    cd hailo_windows_apps/multistream_opencv_cpp &&\
    mkdir build && cd build &&\
    cmake -DMACRO_SET=YOLOV5_APP .. &&\
    cmake --build . --config Release 
