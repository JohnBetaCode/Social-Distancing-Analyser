FROM nvidia/cuda:10.2-cudnn7-devel AS builder

ARG CMAKE_VERSION=3.17.2
ARG OPENCV_VERSION=4.3.0
ARG YOLO_WEIGHTS_URL=https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights 
ARG JOBS_MAKE=4

# Requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    python3-dev \
    libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Get pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    python3 -m pip install numpy && \
    rm -rf get-pip.py

# CMAKE Installation
RUN curl -LOk https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar -xf  cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz -C /opt/ --transform s/cmake-${CMAKE_VERSION}-Linux-x86_64/cmake/ && \
    rm -rf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz

ENV PATH /opt/cmake/bin:$PATH

# OpenCV Installation
RUN curl -LOk https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz && \
    tar -xf ${OPENCV_VERSION}.tar.gz -C /opt --transform s/opencv-${OPENCV_VERSION}/opencv/ && \
    mkdir /opt/opencv/build && cmake -S /opt/opencv -B /opt/opencv/build \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF && \
    make install -C /opt/opencv/build -j${JOBS_MAKE} && \
    rm -rf ${OPENCV_VERSION}.tar.gz

WORKDIR /usr/src/app

# Get YOLOV4
RUN curl -LOk https://github.com/Davidnet/darknet/archive/master.tar.gz && \
    tar -xf master.tar.gz && \
    cd darknet-master && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 && \
    ./build.sh && \
    rm -rf /usr/lib/x86_64-linux-gnu/libcuda.so.1 && cd /usr/src/app \
    rm -rf master.tar.gz && \
    cp darknet-master/libdark.so darknet-master/darknet.py . && \
    mkdir cfg && cp darknet-master/cfg/yolov4.cfg darknet-master/cfg/coco.data cfg/  && \
    mkdir data && cp darknet-master/data/coco.names data/ && \
    curl -LOk ${YOLO_WEIGHTS_URL} && \
    rm -rf darknet-master && \
    mv libdark.so libdarknet.so && \
    sed -i '0,/8/s//64/' cfg/yolov4.cfg

# ToDO(davidnet): Remove after test
RUN curl -Lk -o test.jpg https://www.pasionfutbol.com/__export/1581793721580/sites/pasionlibertadores/img/2020/02/15/leo-messi-barcelona-asistencias-mas-que-goles_crop1581792929892.jpg_423682103.jpg

RUN python3 -m pip install pyaml opencv-python

COPY darknet_detector.py .
COPY python_utils .
COPY social_distancing.py .

ENTRYPOINT ["/bin/bash", "-c", "source configs/env.sh && python3 social_distancing.py"]