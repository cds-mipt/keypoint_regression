ARG CUDA_VERSION=11.3.1
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ARG MAKEFLAGS=-j$(nproc)

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx lsb-release curl\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ROS Noetic installation
ARG ROS_PKG=ros_base
ARG ROS_DISTRO=noetic
ENV ROS_DISTRO=${ROS_DISTRO}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

# ROS Noetic installation
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install \
        --yes \
        --no-install-recommends \
            ros-${ROS_DISTRO}-ros-base \
            ros-${ROS_DISTRO}-cv-bridge \
            ros-${ROS_DISTRO}-vision-opencv \
            ros-${ROS_DISTRO}-tf \
            python3-rosdep \
            python3-rosinstall \
            python3-rosinstall-generator \
            python3-wstool \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install \
    --yes \
    --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-wheel \
        python3-setuptools


# Install xtcocotools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install cython tqdm xtcocotools

# Install torch
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install MMCV
RUN pip install --no-cache-dir pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install MMDet
RUN git clone https://github.com/open-mmlab/mmdetection.git /home/mmdetection
WORKDIR /home/mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -v -e .

# Install MMPose
RUN git clone https://github.com/open-mmlab/mmpose.git /home/mmpose
WORKDIR /home/mmpose
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -v -e .

# Install Original YOLOX
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git /home/yolox
WORKDIR /home/yolox
RUN pip install --no-cache-dir -v -e .

ENV PYTHONPATH="${PYTHONPATH}:/home/yolox:/home/mmdetection:/home/mmpose"