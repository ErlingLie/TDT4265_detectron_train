import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import sys
import time
import os
import json
import argparse

from matplotlib import pyplot as plt


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog




class VideoDetector():
    count = 0
    def __init__(self, video_path, output_dir, model_name, num_classes = 1, model_type = "faster_rcnn_R_50_FPN_3x"):
        self.video_path  = video_path
        self.model_name  = model_name
        self.model_type  = model_type
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            print("Directory " , output_dir ,  " Created ")
        self.num_classes = num_classes
        self.output_path = os.path.join(
                output_dir,
                str(VideoDetector.count) + ".json")
        VideoDetector.count += 1
        self.db = {}
        try:
            with open("label_translation_map.json","r") as json_file:
                self.translation_map = json.load(json_file)
                self.translation_map = {int(k) : int(v) for k,v in self.translation_map.items()}
            print("Using translated label IDs")
        except:
            self.translation_map = {i : i for i in range(self.num_classes)}
            print("Using default label IDs")

    def _setup_bbox_predictor(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+self.model_type+".yaml"))
        cfg.OUTPUT_DIR = "bbox_output"
        cfg.MODEL.WEIGHTS = self.model_name
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.RETINANET.NUM_CLASSES = self.num_classes
        cfg.INPUT.MIN_SIZE_TEST = 0
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2, 3]]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [48, 64], [96, 128], [192, 256], [512, 640]]
        #Initialize a default predictor, the interface to do inference with models in detectron
        return DefaultPredictor(cfg)

    def _add_outputs_to_dict(self, outputs, frame_nmbr):
        '''
        Saves output from current frame to dictionary.
        Format of dictionary is:
        - frame number
            - class_1 : boxes and scores [x0, y0, x1, y1, score]
            - class_2 : boxes and scores [x0, y0, x1, y1, score]
            ...
        Args:
            outputs: Output from detectron2
        '''
        output = outputs["instances"].to('cpu')
        bboxes = output.pred_boxes.tensor.numpy()
        scores = output.scores.numpy().reshape(-1,1)
        arr = np.concatenate((bboxes,scores),axis = 1)
        classes = output.pred_classes.numpy()
        self.db[frame_nmbr] = {}
        for class_id in range(self.num_classes):
            new_class_id = self.translation_map[class_id]
            class_individs = arr[classes == class_id]
            self.db[frame_nmbr][new_class_id] = class_individs.tolist()

    def detect_all_frames(self):
        bbox_predictor = self._setup_bbox_predictor()
        video_cap    = cv2.VideoCapture(self.video_path)
        total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        t0 = time.time()
        i = 0
        ret, frame = video_cap.read()
        if(not ret):
            print("Could not read video " + self.video_path)
        while ret:
            t1 = time.time()
            #Produce some nice console output to show progess
            progress = "\rProgress: " + str(int((i/total_frames)*100)) + " %, fps: " + str(int(i/(t1-t0 + 1.0e-12)))
            print(progress, end="")
            i+=1

            bbox_output  = bbox_predictor(frame)
            self._add_outputs_to_dict(bbox_output, i)
            ret, frame = video_cap.read()

    def save_detections_to_json(self):
        with open(self.output_path, "w") as json_file:
            json.dump(self.db, json_file)
        print("Detections saved to "+ self.output_path + "\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface for running detection on videos")
    parser.add_argument("input", type=str, help = "Directory for input videos")
    parser.add_argument("output", type=str, help = "Directory for output jsons")
    parser.add_argument("--model_name", default="models/model.pth", type=str, help = "Model to use for detection")
    parser.add_argument("--num_classes", default=4, type=int, help="Number of classes in dataset")
    args = parser.parse_args()
    videos = os.listdir(args.input)
    videos.sort()
    for i, video_path in enumerate(videos):
        detector = VideoDetector(os.path.join(args.input, video_path), args.output, args.model_name, args.num_classes)
        print(f"Starting detection on video {i+1}/{len(videos)}: " + video_path)
        detector.detect_all_frames()
        detector.save_detections_to_json()