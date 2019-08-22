import glob
import os
import os.path
import json
import numpy as np
from pycocotools.coco import COCO
import datetime
import collections
import labelme
from labelme._version import __version__
import cv2

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)

def coco2labelme(json_src, json_des):
    jsons = glob.glob("%s/*new.json"%(json_src))
    jsons.sort()
    types = ['ignore', 'road', 'person', 'rider', 'car', 'truck', 'bus', 'motocycle', 'bicycle', 'traffic sign', 'traffic light', 'line', '单实线', '双实线', '单虚线', '双虚线', '左实右虚', '左虚右实', '人行道']
    cIds = ['0', '100','200', '300', '400', '500', '600', '700', '800', '5000', '6000', '7000', '7001', '7002', '7003', '7004', '7005', '7006', '7007']
    for json_f in jsons:
        coco = COCO(json_f)
        img = coco.loadImgs(0)[0]
        catIds = coco.getCatIds(catNms=types)
        annIds = coco.getAnnIds(imgIds=[img["id"]], catIds=catIds)
        anns = coco.loadAnns(annIds)
        d_json_f = os.path.join(json_des, os.path.basename(json_f).replace("_new.json", ".json"))   
        with open(d_json_f, 'w') as d_f:
            obj = {}
            obj["version"] = __version__
            obj["flags"] = {}
            obj["lineColor"] = [0, 255, 0, 128]
            obj["fillColor"] = [255, 0, 0, 128]
            obj["imagePath"] = img["file_name"]
            obj["imageWidth"] = img["width"]
            obj["imageHeight"] = img["height"]     
            obj["imageData"] = None
            shapes = []
            for j, ann in enumerate(anns):
                shape = {}
                shape["label"] = types[cIds.index(str(ann["category_id"]))]
                shape["line_color"] = None
                shape["fill_color"] = None
                segmentation = ann["segmentation"][0]
                if len(segmentation)%2 == 1:
                    print("points error: %s %d"%(json_f, ann["id"]))
                    continue
                points = np.asarray(segmentation).reshape(len(segmentation)//2, 2).tolist()
                if len(points) < 3:
                    print("points size < 3: %s %d"%(json_f, ann["id"]))
                    continue
                shape["points"] = points
                shape["shape_type"] = "polygon"
                flags = {}
                if ann["iscrowd"] == 1 or ann["category_id"] == 0:
                    flags["iscrowd"] = True
                else:
                    flags["iscrowd"] = False
                shape["flags"] = flags
                shapes.append(shape)
            obj["shapes"] = shapes
            d_f.write(json.dumps(obj, ensure_ascii=False))


def labelme2coco(json_src, json_des):
    jsons = glob.glob("%s/*.json"%(json_src))
    jsons.sort()
    types = ['ignore', 'road', 'person', 'rider', 'car', 'truck', 'bus', 'motocycle', 'bicycle', 'traffic sign', 'traffic light', 'line', '单实线', '双实线', '单虚线', '双虚线', '左实右虚', '左虚右实', '人行道']
    cIds = ['0', '100','200', '300', '400', '500', '600', '700', '800', '5000', '6000', '7000', '7001', '7002', '7003', '7004', '7005', '7006', '7007']
    for image_id, label_file in enumerate(jsons): 
        now = datetime.datetime.now()
        data = dict(
            info = dict(
                description = 'BORUN',
                url = ' ',
                version = '1.0',
                year = now.year,
                contributor = 'BORUN Consortium',
                date_created = now.strftime('%Y/%m/%d'),
            ),
            licenses = [dict(
                url = ' ',
                id = 1,
                name = "License",
            )],
            images=[
                # license, file_name, coco_url, height, width, date_captured, flickr_url, id
            ],
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        out_ann_file = os.path.join(json_des, os.path.basename(label_file).replace(".json", "_new.json"))
        with open(label_file) as f:
            label_data = json.load(f)
        if len(label_data) == 0 or len(label_data["shapes"]) == 0:
            print('json data null:', label_file)
            continue
        data['images'].append(dict(
            license = 1,
            file_name = label_data["imagePath"],
            coco_url = ' ',
            height = label_data["imageHeight"],
            width = label_data["imageWidth"],
            date_captured = ' ',
            flickr_url = ' ',
            id = 0,
        ))

        category_ids = []
        for shape in label_data['shapes']:
            label = shape['label']
            if types.count(label) == 0:
                print("unsigned type: %s %s"%(label, label_file))
                continue
            points = shape['points']
            if len(points) < 3:
                print("point count error: %d %s"%(len(points), label_file))
                continue
            category_id = cIds[types.index(label)]
            if category_ids.count(category_id) == 0:
                category_ids.append(category_id)
                data['categories'].append(dict(
                    supercategory = None,
                    id = int(category_id),
                    name = label,
                ))
            segmentations = []  
            points = np.asarray(points).astype(np.int32).flatten().tolist()
            segmentations.append(points)
            if (shape["flags"] and shape["flags"]["iscrowd"] == True) or int(category_id) == 0:
                iscrowd = 1
            else:
                iscrowd = 0
            data["annotations"].append(dict(
                id = len(data['annotations']),
                image_id = 0,
                category_id = int(category_id),
                segmentation = segmentations,
                area = 0,
                bbox = [0,0,0,0],
                iscrowd = iscrowd,
            ))
        with open(out_ann_file, 'w') as wf:
            json.dump(data, wf)
        
def testCoco(json_src, image_src):
    jsons = glob.glob("%s/*new.json"%(json_src))
    jsons.sort()
    types = ['ignore', 'road', 'person', 'rider', 'car', 'truck', 'bus', 'motocycle', 'bicycle', 'traffic sign', 'traffic light', 'line', '单实线', '双实线', '单虚线', '双虚线', '左实右虚', '左虚右实', '人行道']
    cIds = ['0', '100','200', '300', '400', '500', '600', '700', '800', '5000', '6000', '7000', '7001', '7002', '7003', '7004', '7005', '7006', '7007']
    for json_f in jsons:
        coco = COCO(json_f)
        img = coco.loadImgs(0)[0]
        catIds = coco.getCatIds(catNms=types)
        annIds = coco.getAnnIds(imgIds=[img["id"]], catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)
        img_f = os.path.join(image_src, os.path.basename(json_f).replace("_new.json", ".jpg"))
        img = cv2.imread(img_f)
        for j, ann in enumerate(anns):
            m = coco.annToMask(ann) > 0
            img[m] = np.random.randint(0, 255, 3).tolist()
        cv2.imshow("image",img)
        key = cv2.waitKey(0)
        if key == ord(' '):
            continue
        if key == ord('q'):
            break


if __name__ == "__main__":
    testCoco("/home/team/data/gene", "/home/team/data/data")