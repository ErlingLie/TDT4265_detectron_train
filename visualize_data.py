#visualize training data
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import cv2
register_coco_instances("traffic", {}, "code_workspace/datasets/train_split.json",
                        "code_workspace/datasets/train/images")

MetadataCatalog.get("traffic").thing_colors = [(0, 255, 0), (0, 0, 255), (255,0,0), (127, 127, 255)]
MetadataCatalog.get("traffic").thing_classes = ["vehicle", "person", "sign", "cyclist"]
my_dataset_train_metadata = MetadataCatalog.get("traffic")
dataset_dicts = DatasetCatalog.get("traffic")

import random
from detectron2.utils.visualizer import Visualizer
for i, d in enumerate(random.sample(dataset_dicts, 3)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("code_workspace/image" + str(i) + ".jpg", vis.get_image()[:, :, ::-1])