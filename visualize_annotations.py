import json
from matplotlib import pyplot as plt
import numpy as np
import cv2


color_map = [(0,0,255), (255,0,0), (0,255,0), (127, 127, 54)]
print("Making video")
#file_path = "bbox_output/coco_instances_results.json"
file_path = "code_workspace/datasets/train_master.json"
with open(file_path) as json_file:
    data = json.load(json_file)
    data = data["annotations"]
    path_to_output = "code_workspace/annotations.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(
            path_to_output, 
            fourcc, 
            30.0, 
            (1280, 960),
            True
            )
    im_id = data[1000]["image_id"]
    im = plt.imread("code_workspace/datasets/train/images/"+ str(im_id) + ".jpg")
    detections = []
    for i, val in enumerate(data[1000:]):
        if(i >3000):
            break
        if(val["image_id"]-im_id>0):
            while(im_id != val["image_id"]):
                print(f"Showing image {im_id}")
                video_out.write(im)
                im_id = val["image_id"]
                im = plt.imread("code_workspace/datasets/train/images/" + str(im_id) + ".jpg")
        else:
            bbox = val["bbox"]
            bbox = np.array(bbox)
            bbox[2:4] += bbox[0:2] #Convert [x1,y1,w,h] to [x1,y1,x2,y2]
            tl = (int(bbox[0]), int(bbox[1]))
            br = (int(bbox[2]),int(bbox[3]))
            cv2.rectangle(im, tl, br, color_map[val["category_id"]-1], thickness = 3)

    video_out.release()