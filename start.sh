#!/bin/bash

docker build . --tag keypoints_image
docker create --gpus all -it -p 1001:1001 --ipc=host -v /home/ilyabasharov/workspace/keypoint_regression:/home/keypoint_regression --name=keypoints_container keypoints_image /bin/bash
docker start keypoints_container
docker exec -it keypoints_container /bin/bash
