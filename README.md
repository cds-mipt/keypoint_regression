# Door handle keypoint regression

## Before installation
Change the script `start.sh` with correct paths to your code and other params.

## Install framework
```bash
bash start.sh
# now you are inside docker container
bash # to change default command manager
source /opt/ros/noetic/setup.bash # to activate ROS
export PYTHONPATH="${PYTHONPATH}:/home/yolox:/home/mmdetection:/home/mmpose" # to activate installed libs
```

Download rosbags from [here](https://drive.google.com/drive/folders/1IVeVQ_eA6Fvj4WNRMeF381JNXRccRIyF?usp=sharing) and weights from [here](https://drive.google.com/drive/folders/17Lj1PnW_B3EiseFtC0K00Sd1L4LimIVI?usp=sharing)

## To run in on single image 
```bash
python3 tools/rosbag_eval.py image \
    --path assets/000.jpg \ # put your image path here 
    --save_result \
    --name yolox-s \
    --exp_file_det src/detector/config.py \
    --ckpt_det weights/yolox-s_door_handle.pth \
    --device_det cuda:0 \
    --conf 0.1 \
    --exp_file_reg src/regressor/config.py \
    --ckpt_reg weights/hrnet_w32_256x192_door_handle.pth \
    --device_reg cuda:0
```

## To run in on ros bag 
```bash
rosbag play data/ros_bags/2022-07-06-08-37-01-002.bag # as an example
# needed topics inside a rosbag:
# 1) /realsense_back/color/image_raw
# 2) /realsense_back/aligned_depth_to_color/image_raw
# 3) /realsense_back/color/camera_info
# the result should be compared with topic:
# 1) /door_handle
python3 tools/rosbag_eval.py rosbag \
    --save_result \
    --name yolox-s \
    --exp_file_det src/detector/config.py \
    --ckpt_det weights/yolox-s_door_handle.pth \
    --device_det cuda:0 \
    --conf 0.7 \
    --exp_file_reg src/regressor/config.py \
    --ckpt_reg weights/hrnet_w32_256x192_door_handle.pth \
    --device_reg cuda:0
```
