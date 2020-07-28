# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
import os
import subprocess as sup
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, launch
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_test_loader
import sys
import random
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
# Extra utils
# prep dataset

register_coco_instances("train_set", {}, "code_workspace/datasets/train_split.json",
                        "code_workspace/datasets/train/images")

register_coco_instances("val_set", {}, "code_workspace/datasets/val_split.json",
                        "code_workspace/datasets/train/images")
register_coco_instances("test_set", {}, "code_workspace/datasets/test_labels_min.json",
                        "code_workspace/datasets/test/images")

MetadataCatalog.get(
    "train_set").json_file = "code_workspace/datasets/train_split.json"

# train

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))


cfg.DATASETS.TRAIN = ("train_set",)
cfg.DATASETS.TEST = ("val_set",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
print(cfg.MODEL.WEIGHTS)

cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (9000, 9500)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.MASK_ON = False
cfg.TEST.EVAL_PERIOD = 2000

cfg.INPUT.MIN_SIZE_TEST = 0
cfg.INPUT.CROP.ENABLED =True
cfg.OUTPUT_DIR = "code_workspace/output_low_lr"


class TrainerWithEval(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)



trainer = TrainerWithEval(cfg)
trainer.resume_or_load(resume=False)

trainer.train()
