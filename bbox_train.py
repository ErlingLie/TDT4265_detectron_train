print("Importing")
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import sys
import subprocess as sup
import os
import time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

#register datasets
register_coco_instances("train_set", {}, "code_workspace/datasets/train_split.json",
                        "code_workspace/datasets/train/images")

register_coco_instances("val_set", {}, "code_workspace/datasets/val_split.json",
                        "code_workspace/datasets/train/images")
register_coco_instances("test_set", {}, "code_workspace/datasets/test_labels_min.json",
                        "code_workspace/datasets/test/images")




#train

#model alias
faster_rcnn = "faster_rcnn_R_101_FPN_3x"
retinanet = "retinanet_R_50_FPN_3x"

#Create a parameter grid to enable tuning hyperparameters, works slowly without enabling parallellism
par_grid = {
        "NETWORK": [faster_rcnn, retinanet],
        "BASE_LR": [0.01], 
        "MAX_ITER": [10000],
        "STEPS": [(8000, 9000)],
        "BATCH_SIZE_PR_IMAGE": [512], 
        "MOMENTUM": [0.9], 
        "IMS_PER_BATCH": [8],
        "Crop" : [False, True]
        }



#After training, find the n best performing networks conncerning AP, input is list of configs,
#list of AP dictionaries, and the number of networks we want
def find_n_best_results(cfgs, aps, n):
    tuples = []
    
    for tup in zip(aps, cfgs):
        new_tup = (tup[0]["bbox"]["AP-Lobster"], tup[1])
        tuples.append(new_tup)
    tuples.sort()
    i = 0
    max_len = len(tuples)-1
    while i<n and max_len-i>=0:
        with open("./code_workspace/grid_results.txt", "a") as f:
            f.write(f"Network number: {i+1}\n\n")
            f.write(f"Lobster AP: {tuples[max_len - i][0]}\n")
            f.write(f"Config:\n {tuples[max_len-i][1]}\n\n\n")
            f.write("_"*15)
        print()
        print("Network number ", i+1)
        print()
        print("Lobster AP:", tuples[max_len-i][0])
        print("Config:")
        print(tuples[max_len-i][1])
        i+=1

#Generate different combinations of configs with the given hyperparameters and networks, output is a list of cfg objects
def generate_cfgs(grid):
    cfgs = []
    
    for it in grid["MAX_ITER"]:
        for bat_size_per_img in grid["BATCH_SIZE_PR_IMAGE"]:
            for momentum in grid["MOMENTUM"]:
                for ims_per_bat in grid["IMS_PER_BATCH"]:
                    for lr in grid["BASE_LR"]:
                        for network in grid["NETWORK"]:
                            for step in grid["STEPS"]:

                                cfg = get_cfg()
                                cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+network+".yaml"))
                                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/"+network+".yaml")
                                cfg.DATASETS.TRAIN = ("train_set", )
                                cfg.DATASETS.TEST = ("val_set", )
                                cfg.DATALOADER.NUM_WORKERS = 4
                                cfg.SOLVER.IMS_PER_BATCH = ims_per_bat
                                cfg.SOLVER.BASE_LR = lr
                                cfg.SOLVER.MAX_ITER = it
                                cfg.SOLVER.MOMENTUM = momentum
                                cfg.SOLVER.STEPS = step
                                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bat_size_per_img
                                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
                                cfg.MODEL.MASK_ON = False
                                cfg.MODEL.RETINANET.NUM_CLASSES = 4
                                cfg.OUTPUT_DIR = "./code_workspace/output/grid_search"
                                cfg.INPUT.MIN_SIZE_TEST = 0
                                cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2, 3]]
                                cfgs.append(cfg)

    return cfgs

#Start the training with detectron2 default trainer
def train(cfgs):
    results = []
    for cfg in cfgs:
        print("Starting training with cfg: ", cfg)
        trainer = DefaultTrainer(cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
    
        evaluator = COCOEvaluator("lob_tst", cfg, False, cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, "lob_tst")
        results.append(inference_on_dataset(trainer.model, val_loader, evaluator))
        with open("./code_workspace/Grid_log.txt", "a") as f:
            f.write(f"Lobster AP: {results[-1]}\n")
            f.write(f"Config:\n {cfg}\n\n\n")
            f.write("_"*15)

    return results

cfgs = generate_cfgs(par_grid)
aps = train(cfgs)
find_n_best_results(cfgs, aps, n=5)

