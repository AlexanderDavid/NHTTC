FROM ros:noetic-ros-base

# install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-tf2 \
    libeigen3-dev \
    git

RUN mkdir -p /opt/nhttc_ws/src/NHTTC
WORKDIR /opt/nhttc_ws/src
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_init_workspace"