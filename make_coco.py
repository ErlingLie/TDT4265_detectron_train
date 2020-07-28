import json

def make_categories(str_list):
    categories = []
    for i, string in enumerate(str_list):
        category = {}
        category["name"] = string
        category["id"] = i+1
        category["supercategory"] = ""
        categories.append(category)
    return categories

def make_license():
    empty_license = {"name": "", "id": 0, "url": ""}
    return [empty_license]


def make_annotation(bbox, image, category, ann_id):
    annotation = {"bbox": [], "image_id" : image,
                    "category_id" : category + 1, "id" : ann_id,
                    "segmentation" : [], "iscrowd" : 0}
    xtl = bbox[0]
    ytl = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    annotation["bbox"] = [xtl, ytl, w, h]
    annotation["area"] = w*h

    return annotation

def make_image(image_id):
    image = {"id" : image_id, "width" : 1280,
            "height" : 960, "file_name": str(image_id) + ".jpg" }
    return image

if __name__ == "__main__":
    test = {"annotations" : [], "images" : [], "licences" : make_license(), "categories" : []}
    train = {"annotations" : [], "images" : [], "licences" : make_license(), "categories" : []}
    test["categories"] = make_categories(["vehicle", "person", "sign", "cyclist"])
    train["categories"] = make_categories(["vehicle", "person", "sign", "cyclist"])
    with open("labels_test.json", "r") as json_file:
        db = json.load(json_file)
    train_id = 1
    test_id = 1
    for image in db:
        if not image["annotation_completed"]:
            continue
        is_test = image["is_test"]
        image_id = image["image_id"]
        if image_id % 7 != 0:
            continue
        if(is_test):
            test["images"].append(make_image(image_id))
            for annotation in image["bounding_boxes"]:
                bbox = [annotation["xmin"], annotation["ymin"], annotation["xmax"],annotation["ymax"]]
                label = annotation["label_id"]
                test["annotations"].append(make_annotation(bbox, image_id, label, test_id))
                test_id += 1
        else:
            train["images"].append(make_image(image_id))
            for annotation in image["bounding_boxes"]:
                bbox = [annotation["xmin"], annotation["ymin"], annotation["xmax"],annotation["ymax"]]
                label = annotation["label_id"]
                train["annotations"].append(make_annotation(bbox, image_id, label, train_id))
                train_id += 1
    test["annotations"].sort(key = lambda x : x["image_id"])
    test["images"].sort(key = lambda x : x["id"])
    train["annotations"].sort(key = lambda x : x["image_id"])
    train["images"].sort(key = lambda x : x["id"])
    with open("datasets/test_labels_mini.json", "w") as json_file:
        json.dump(test, json_file)

    with open("datasets/train_master_mini.json", "w") as json_file:
        json.dump(train, json_file)
