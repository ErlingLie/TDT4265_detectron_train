import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import sys
import time
import os


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer

#metadata
MetadataCatalog.get("traffic").thing_colors = [(0, 255, 0), (0, 0, 255), (255,0,0), (127, 127, 255)]
MetadataCatalog.get("traffic").thing_classes = ["vehicle", "person", "sign", "cyclist"]

faster_rcnn = "faster_rcnn_R_50_FPN_3x"


#Produce a test video, with annotations overlayed, make sure network matches the .pth file
def test(path_to_input, path_to_output, network):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+network+".yaml"))
    cfg.OUTPUT_DIR = "./code_workspace/output"
    cfg.MODEL.WEIGHTS = os.path.join(
            cfg.OUTPUT_DIR, 
            "model_0009999.pth"
            )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST   = 0.2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.MASK_ON = False
    cfg.TEST.EVAL_PERIOD = 5000

    #cfg.INPUT.MIN_SIZE_TEST = 0
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2, 3]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32], [48, 64], [96, 128], [192, 256], [512, 640]]

    predictor = DefaultPredictor(cfg)


    video_cap = cv2.VideoCapture(path_to_input)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, i_frame = video_cap.read()
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(
            path_to_output, 
            fourcc, 
            30.0, 
            (int(i_frame.shape[1]), int(i_frame.shape[0])),
            True
            )
    total_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    i=0
    
    #timing
    t0 = time.time()

    #try to not do saves in between iterations, instead save all in list, then do all saving at the end
    out_list = []
    frames = []
    predict_time = 0

    t2 = time.time()
    while True:
        t3 = time.time()
        ret, frame = video_cap.read()
        if (not ret):
            break
        #Produce some nice console output to show progess
        progress = "\r %progress: " + str(int((i/total_frames)*100)) + "    " + "fps: " + str(int(i/(t3-t0))) 
        i+=1
        sys.stdout.write(progress)
        sys.stdout.flush()
        
        t4 = time.time()
        outputs = predictor(frame)
        t5 = time.time()
        predict_time += t5-t4
        out_list.append(outputs["instances"].to("cpu"))
        frames.append(frame)
    t22 = time.time()
    inference_time = t22-t2
    print()
    print("Inference complete, creating video") 
    t10 = time.time()
    for output, frame in zip(out_list, frames):
        v = Visualizer(
                frame,
                MetadataCatalog.get("traffic"),
                scale=1,
                instance_mode=ColorMode.SEGMENTATION)
        #output.remove("scores")
        v = v.draw_instance_predictions(output)
        video_out.write(v.get_image())
    t11 = time.time()
    print("Time to create video: ", t11-t10)

    #timing
    t1 = time.time()
    print("average fps: ", total_frames/inference_time)
    print("total time: ", t1-t0)
    print("%total predict: ", predict_time/(t1-t0))
    print("Video produced on path: ", path_to_output)
    video_out.release()
    video_cap.release()


test(
        "./code_workspace/videos/demo_video2.mp4",
        "./code_workspace/tesla_output2.mp4", 
        faster_rcnn
        )
