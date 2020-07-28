import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import sys
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
#Extra utils

import subprocess as sup
import os
##prep dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train_set", {}, "code_workspace/datasets/train_split.json",
                        "code_workspace/datasets/train/images")

register_coco_instances("val_set", {}, "code_workspace/datasets/val_split.json",
                        "code_workspace/datasets/train/images")
register_coco_instances("test_set", {}, "code_workspace/datasets/test_labels_mini.json",
                        "code_workspace/datasets/test/images")

def make_cfg(weight_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))


    cfg.DATASETS.TRAIN = ("train_set",)
    cfg.DATASETS.TEST = ("test_set",)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS =weight_path

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (8000, 9000)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.MASK_ON = False
    cfg.TEST.EVAL_PERIOD = 5000

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.2

    # cfg.INPUT.MIN_SIZE_TEST = 0
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2, 3]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [48, 64], [96, 128], [192, 256], [512, 640]]

    cfg.OUTPUT_DIR = "code_workspace/output/inference_2"
    return cfg

weights = "./code_workspace/output/model_final.pth"
cfg = make_cfg(weights)
evaluator = COCOEvaluator("test_set", cfg, False, cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "test_set")
predictor = DefaultPredictor(cfg)
print("Running inference with weights: " + weights)
inference_on_dataset(predictor.model, val_loader, evaluator)
