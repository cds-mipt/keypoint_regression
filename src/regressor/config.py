checkpoint_config = dict(interval=10)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author=
        'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/'),
    keypoint_info=dict({
        0:
        dict(name='0', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(name='1', id=1, color=[51, 153, 255], type='upper', swap=''),
        2:
        dict(name='2', id=2, color=[51, 153, 255], type='upper', swap=''),
        3:
        dict(name='3', id=3, color=[51, 153, 255], type='upper', swap=''),
        4:
        dict(name='4', id=4, color=[51, 153, 255], type='upper', swap=''),
        5:
        dict(name='5', id=5, color=[51, 153, 255], type='upper', swap='')
    }),
    skeleton_info=dict({
        0:
        dict(link=('0', '1'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('1', '2'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('2', '3'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('3', '4'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('4', '5'), id=4, color=[0, 255, 0])
    }),
    joint_weights=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 
    ])
evaluation = dict(interval=10, metric='PCK', save_best='PCK')
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    step=[17, 35])
total_epochs = 40
channel_cfg = dict(
    num_output_channels=6,
    dataset_joints=6,
    dataset_channel=[[
        0, 1, 2, 3, 4, 5, 
    ]],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 
    ])
model = dict(
    type='TopDown',
    pretrained=
    'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=6,
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))
data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=6,
    num_joints=6,
    dataset_channel=[[
        0, 1, 2, 3, 4, 5, 
    ]],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 
    ],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file=
    'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ])
]
data_root = 'data/coco_tiny'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16),
    train=dict(
        type='TopDownCOCOTinyDataset',
        ann_file='train.txt',
        img_prefix='/home/keypoints/Doors/VOC2007',
        data_cfg=dict(
            image_size=[192, 256],
            heatmap_size=[48, 64],
            num_output_channels=6,
            num_joints=6,
            dataset_channel=[[
                0, 1, 2, 3, 4, 5, 
            ]],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 
            ],
            soft_nms=False,
            nms_thr=1.0,
            oks_thr=0.9,
            vis_thr=0.2,
            use_gt_bbox=False,
            det_bbox_thr=0.0,
            bbox_file=
            'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownRandomFlip', flip_prob=0.5),
            dict(
                type='TopDownHalfBodyTransform',
                num_joints_half_body=8,
                prob_half_body=0.3),
            dict(
                type='TopDownGetRandomScaleRotation',
                rot_factor=40,
                scale_factor=0.5),
            dict(type='TopDownAffine'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(type='TopDownGenerateTarget', sigma=2),
            dict(
                type='Collect',
                keys=['img', 'target', 'target_weight'],
                meta_keys=[
                    'image_file', 'joints_3d', 'joints_3d_visible', 'center',
                    'scale', 'rotation', 'bbox_score', 'flip_pairs'
                ])
        ],
        dataset_info=dict(
            dataset_name='coco',
            paper_info=dict(
                author=
                'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(name='0', id=0, color=[51, 153, 255], type='upper', swap=''),
                1:
                dict(name='1', id=1, color=[51, 153, 255], type='upper', swap=''),
                2:
                dict(name='2', id=2, color=[51, 153, 255], type='upper', swap=''),
                3:
                dict(name='3', id=3, color=[51, 153, 255], type='upper', swap=''),
                4:
                dict(name='4', id=4, color=[51, 153, 255], type='upper', swap=''),
                5:
                dict(name='5', id=5, color=[51, 153, 255], type='upper', swap='')
            }),
            skeleton_info=dict({
                0:
                dict(link=('0', '1'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('1', '2'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('2', '3'), id=2, color=[0, 255, 0]),
                3:
                dict(link=('3', '4'), id=3, color=[0, 255, 0]),
                4:
                dict(link=('4', '5'), id=4, color=[0, 255, 0])
            }),
            joint_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 
            ])),
    val=dict(
        type='TopDownCOCOTinyDataset',
        ann_file='val.txt',
        img_prefix='/home/keypoints/Doors/VOC2007',
        data_cfg=dict(
            image_size=[192, 256],
            heatmap_size=[48, 64],
            num_output_channels=6,
            num_joints=6,
            dataset_channel=[[
                0, 1, 2, 3, 4, 5, 
            ]],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 
            ],
            soft_nms=False,
            nms_thr=1.0,
            oks_thr=0.9,
            vis_thr=0.2,
            use_gt_bbox=False,
            det_bbox_thr=0.0,
            bbox_file=
            'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownAffine'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'image_file', 'center', 'scale', 'rotation', 'bbox_score',
                    'flip_pairs'
                ])
        ],
        dataset_info=dict(
            dataset_name='coco',
            paper_info=dict(
                author=
                'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(name='0', id=0, color=[51, 153, 255], type='upper', swap=''),
                1:
                dict(name='1', id=1, color=[51, 153, 255], type='upper', swap=''),
                2:
                dict(name='2', id=2, color=[51, 153, 255], type='upper', swap=''),
                3:
                dict(name='3', id=3, color=[51, 153, 255], type='upper', swap=''),
                4:
                dict(name='4', id=4, color=[51, 153, 255], type='upper', swap=''),
                5:
                dict(name='5', id=5, color=[51, 153, 255], type='upper', swap='')
            }),
            skeleton_info=dict({
                0:
                dict(link=('0', '1'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('1', '2'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('2', '3'), id=2, color=[0, 255, 0]),
                3:
                dict(link=('3', '4'), id=3, color=[0, 255, 0]),
                4:
                dict(link=('4', '5'), id=4, color=[0, 255, 0])
            }),
            joint_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 
            ])),
    test=dict(
        type='TopDownCOCOTinyDataset',
        ann_file='val.txt',
        img_prefix='/home/keypoints/Doors/VOC2007',
        data_cfg=dict(
            image_size=[192, 256],
            heatmap_size=[48, 64],
            num_output_channels=6,
            num_joints=6,
            dataset_channel=[[
                0, 1, 2, 3, 4, 5, 
            ]],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 
            ],
            soft_nms=False,
            nms_thr=1.0,
            oks_thr=0.9,
            vis_thr=0.2,
            use_gt_bbox=False,
            det_bbox_thr=0.0,
            bbox_file=
            'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json'
        ),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='TopDownAffine'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'image_file', 'center', 'scale', 'rotation', 'bbox_score',
                    'flip_pairs'
                ])
        ],
        dataset_info=dict(
            dataset_name='coco',
            paper_info=dict(
                author=
                'Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence',
                title='Microsoft coco: Common objects in context',
                container='European conference on computer vision',
                year='2014',
                homepage='http://cocodataset.org/'),
            keypoint_info=dict({
                0:
                dict(name='0', id=0, color=[51, 153, 255], type='upper', swap=''),
                1:
                dict(name='1', id=1, color=[51, 153, 255], type='upper', swap=''),
                2:
                dict(name='2', id=2, color=[51, 153, 255], type='upper', swap=''),
                3:
                dict(name='3', id=3, color=[51, 153, 255], type='upper', swap=''),
                4:
                dict(name='4', id=4, color=[51, 153, 255], type='upper', swap=''),
                5:
                dict(name='5', id=5, color=[51, 153, 255], type='upper', swap='')
            }),
            skeleton_info=dict({
                0:
                dict(link=('0', '1'), id=0, color=[0, 255, 0]),
                1:
                dict(link=('1', '2'), id=1, color=[0, 255, 0]),
                2:
                dict(link=('2', '3'), id=2, color=[0, 255, 0]),
                3:
                dict(link=('3', '4'), id=3, color=[0, 255, 0]),
                4:
                dict(link=('4', '5'), id=4, color=[0, 255, 0])
            }),
            joint_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
            ],
            sigmas=[
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 
            ])))
work_dir = 'work_dirs/hrnet_w32_coco_tiny_256x192'
gpu_ids = range(0, 1)
seed = 0
