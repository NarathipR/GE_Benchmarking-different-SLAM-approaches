# Use Ubuntu 14.04 as the base image
FROM ubuntu:14.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    sudo \
    lsb-release \
    wget \
    gnupg2

# Set up the ROS repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add the ROS repository key
RUN wget http://packages.ros.org/ros.key -O - | apt-key add -

# Install ROS Indigo
RUN apt-get update && apt-get install -y \
    ros-indigo-desktop-full

# Initialize rosdep
RUN rosdep init && rosdep update

# Source the ROS setup.bash script
RUN echo "source /opt/ros/indigo/setup.bash" >> /etc/bash.bashrc

# Install dependencies for building ROS packages
RUN apt-get install -y \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential \
    git \
    cmake \
    python-catkin-tools

# Install additional dependencies
RUN apt-get install -y \
    ros-indigo-libg2o \
    ros-indigo-cv-bridge \
    liblapack-dev \
    libblas-dev \
    freeglut3-dev \
    libqglviewer-dev \
    libsuitesparse-dev \
    libx11-dev

# Set the working directory
WORKDIR /workspace

# Create and initialize the ROS build workspace
# RUN mkdir -p /workspace/rosbuild_ws/package_dir && \
#     cd /workspace/rosbuild_ws && \
#     rosws init . /opt/ros/indigo && \
#     rosws set /workspace/rosbuild_ws/package_dir -t . && \
#     echo "source /workspace/rosbuild_ws/setup.bash" >> ~/.bashrc

# Set the default command to run when starting the container
CMD ["bash"]
