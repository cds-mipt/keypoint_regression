#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import csv
import time
from loguru import logger

import cv2
import numpy as np

import torch
import tqdm

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import (
    fuse_model,
    get_model_info,
    postprocess,
)

from mmpose.apis import (
    init_pose_model,
    inference_top_down_pose_model,
    vis_pose_result,
)

import matplotlib.pyplot as plt

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
ROSBAG_EXT = [".bag", ]
DOOR_HANDLE_CLASSES = ["bb", ]


def make_parser():
    parser = argparse.ArgumentParser("Door handle demo!")
    parser.add_argument(
        "demo",
        default="image",
        help="demo type, eg. image, video, webcam or rosbag",
    )
    parser.add_argument(
        "-expn", "--experiment-name",
        type=str,
        default=None,
        help="experiment name",
    )
    parser.add_argument(
        "--path",
        default="./assets/dog.jpg",
        help="path to images or video",
    )
    parser.add_argument(
        "--camid",
        type=int,
        default=0,
        help="webcam demo camera id",
    )
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    #detector config
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="YoloX model name. See supported names on github.com/Megvii-BaseDetection/YOLOX.git",
    )
    parser.add_argument(
        "-fd", "--exp_file_det",
        default=None,
        type=str,
        help="Path to YoloX config file",
    )
    parser.add_argument(
        "-cd", "--ckpt_det",
        default=None,
        type=str,
        help="ckpt YoloX for eval",
    )
    parser.add_argument(
        "--device_det",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument(
        "--conf",
        default=0.3,
        type=float,
        help="test confidence threshold",
    )
    parser.add_argument(
        "--nms",
        default=0.3,
        type=float,
        help="test NMS threshold",
    )
    parser.add_argument(
        "--tsize",
        default=None,
        type=int,
        help="test img size",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    #keypoint config
    parser.add_argument(
        "-cr", "--ckpt_reg",
        default=None,
        type=str,
        help="ckpt HRNet-based model for eval",
    )
    parser.add_argument(
        "-fr", "--exp_file_reg",
        default=None,
        type=str,
        help="Path to HRNet-based config",
    )
    parser.add_argument(
        "--device_reg",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=DOOR_HANDLE_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = [
            {
                'bbox': np.array(
                    bboxes[i].detach().cpu().numpy().tolist() + [scores[i].item(), ]
                )
            }
            for i in range(len(bboxes))
            if scores[i] > cls_conf
        ]

        return vis_res


def image_demo(predictor, regressor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in tqdm.tqdm(files):
        outputs, img_info = predictor.inference(image_name)
        result_det = predictor.visual(outputs[0], img_info, predictor.confthre)
        pose_results, returned_outputs = inference_top_down_pose_model(
            regressor,
            img_info['raw_img'],
            result_det,
            bbox_thr=0.0,
            format='xyxy',
            dataset='TopDownCocoDataset',
        )
        result_image = vis_pose_result(
            regressor,
            img_info['raw_img'],
            pose_results,
            kpt_score_thr=0.5,
            show=False,
        )
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

def rosbag_demo(
    predictor, regressor, vis_folder, path, current_time, save_result):
    
    if save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        logger.info(f"video save_path is {save_folder}")
    bag = rosbag.Bag(path, 'r')
    bridge = CvBridge()
    images = []
    for topic, msg, t in bag.read_messages(topics=['/realsense_back/color/image_raw', ]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        images.append(cv_img.copy())
    depths = []
    for topic, msg, t in bag.read_messages(topics=['/realsense_back/aligned_depth_to_color/image_raw', ]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        depths.append(cv_img.copy())
    assert len(images) == len(depths), 'lens of images and depths should be the same.'
    for i, (image, depth) in enumerate(tqdm.tqdm(zip(images, depths))):
        outputs, img_info = predictor.inference(image)
        result_det = predictor.visual(outputs[0], img_info, predictor.confthre)
        pose_results, returned_outputs = inference_top_down_pose_model(
            regressor,
            img_info['raw_img'],
            result_det,
            bbox_thr=0.0,
            format='xyxy',
            dataset='TopDownCocoDataset',
        )
        result_image = vis_pose_result(
            regressor,
            img_info['raw_img'],
            pose_results,
            kpt_score_thr=0.,
            show=False,
        )
        if save_result:
            save_file_name = os.path.join(save_folder, f'{i}_image.png')
            cv2.imwrite(save_file_name, result_image)
            save_file_name = os.path.join(save_folder, f'{i}_depth.jpg')
            plt.imsave(save_file_name, depth)

def imageflow_demo(predictor, regressor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
            
class ProcessRosBag:
    
    def __init__(self, predictor, regressor, save_folder, save_result):
        self.predictor = predictor
        self.regressor = regressor
        self.save_folder = save_folder
        self.save_result = save_result
        
        from message_filters import (
            Subscriber,
            TimeSynchronizer,
        )
        from sensor_msgs.msg import (
            Image,
            CameraInfo,
        )
        from cv_bridge import CvBridge
        from geometry_msgs.msg import PointStamped
        import rospy
        import tf
        
        rospy.init_node('keypoints_detection')
        self.listner = tf.TransformListener()
        
        self.synchronizer = TimeSynchronizer(
            [
                Subscriber('/realsense_back/color/image_raw', Image),
                Subscriber('/realsense_back/aligned_depth_to_color/image_raw', Image),
                #Subscriber('/door_handle', PointStamped),
                Subscriber('/realsense_back/color/camera_info', CameraInfo),
            ],
            queue_size=500,
        )
        self.synchronizer.registerCallback(self.rosbag_process)
        self.publisher = rospy.Publisher(
            '/3d_handle_coordinates',
            PointStamped,
            queue_size=500,
        )
        
        self.bridge = CvBridge()
        self.count = 0
        
    def depth_to_pc(
        self,
        depth,
        keypoints_coordinates,
        camera_info,
    ):
        depth = depth / 1000
        fx, _, cx, _, fy, cy, _, _, _ = camera_info
        
        height, width = depth.shape
        
        for i in range(len(keypoints_coordinates)):
            two_near_center_pixels = keypoints_coordinates[i]['keypoints'][[3, 4], :2]
            keypoints_coordinates[i]['middle_keypoint'] = np.mean(two_near_center_pixels, axis=0)
            two_near_center_image = []
            
            check_flag = True
            
            for point_coordinates_pixels in two_near_center_pixels:
                
                base_y = int(point_coordinates_pixels[1])
                base_x = int(point_coordinates_pixels[0])
                
                check_dims =  0 <= base_x <= width \
                    and 0 <= base_y <= height
                
                if not check_dims:
                    logger.info(
                        f'point is outside the frame dimension, 0 <= {base_x} <= {width}, 0 <= {base_y} <= {height}'
                    )
                    check_flag = False
                    break
                    
                mean_approx_by_deltas = [depth[base_y, base_x], ]
                
                for dx in (1, -1):
                    for dy in (1, -1):
                        new_x = base_x + dx
                        new_y = base_y + dy
                        
                        new_point_inside = (0 <= new_x <= width) and (0 <= new_y <= height)
                        
                        if not new_point_inside:
                            continue
                        
                        mean_approx_by_deltas.append(depth[new_y, new_x])
                        
                z = np.mean(mean_approx_by_deltas)
                
                two_near_center_image.append(
                    [
                        z * (base_x - cx) / fx,
                        z * (base_y - cy) / fy,
                        z,
                    ]
                )
            
            middle_point = np.mean(two_near_center_image, axis=0)
            
            if check_flag:
                keypoints_coordinates[i]['keypoints_3d'] = middle_point
            else:
                keypoints_coordinates[i]['keypoints_3d'] = None
            
    def rosbag_process(
        self,
        image_msg,
        depth_msg,
        #point,
        camera_info_msg,
    ):
        cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        #gt_projected_point = self.listner.transformPoint("rs_camera", point)
        
        assert cv_img.shape[:2] == cv_depth.shape, 'Shapes of depth and image should be the same'
        
        outputs, img_info = self.predictor.inference(cv_img)
        result_det = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.regressor,
            img_info['raw_img'],
            result_det,
            bbox_thr=0.0,
            format='xyxy',
            dataset='TopDownCocoDataset',
        )
        
        result_image = vis_pose_result(
            self.regressor,
            img_info['raw_img'],
            pose_results,
            kpt_score_thr=0.,
            show=False,
        )
            
        self.depth_to_pc(
            depth=cv_depth,
            keypoints_coordinates=pose_results,
            camera_info=camera_info_msg.K,
        )
        
        if self.save_result:
            
            os.makedirs(os.path.join(self.save_folder, 'images'), exist_ok=True)
            save_file_name = os.path.join(self.save_folder, 'images', f'{self.count}_image.png')
            cv2.imwrite(save_file_name, result_image)
            os.makedirs(os.path.join(self.save_folder, 'depths'), exist_ok=True)
            save_file_name = os.path.join(self.save_folder, 'depths', f'{self.count}_depth.jpg')
            plt.imsave(save_file_name, cv_depth)
            os.makedirs(os.path.join(self.save_folder, 'coordinates'), exist_ok=True)
            save_file_name = os.path.join(self.save_folder, 'coordinates', f'{self.count}_coord.csv')
            
            keys = pose_results[0].keys()
            
            with open(save_file_name, 'w', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(pose_results)
        
        self.count += 1
        
    def run(
        self,
    ):
        import rospy
        rospy.spin()

        
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device_det == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt_det is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt_det
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, DOOR_HANDLE_CLASSES, trt_file, decoder,
        args.device_det, args.fp16, args.legacy,
    )

    regressor = init_pose_model(
        args.exp_file_reg, args.ckpt_reg, args.device_reg,
    )

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, regressor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, regressor, vis_folder, current_time, args)
    elif args.demo == "rosbag":
        process_ros = ProcessRosBag(
            predictor=predictor,
            regressor=regressor,
            save_folder=vis_folder,
            save_result=args.save_result,
        )
        
        process_ros.run()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file_det, args.name)

    main(exp, args)
