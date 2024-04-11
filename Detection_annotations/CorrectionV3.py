# Correction V2

"""
Description:

This script is written to correct the Emotic Dataset annotations using the DETR model as a SOA model for correction in order to improve  BeNeT Scores.

@author: Jalal
"""

# libraries
import torch
import numpy as np
import torchvision.transforms as T 
from PIL import Image
from torchvision.transforms import functional as F
import os
import json
from transformers import AutoImageProcessor, DetrForObjectDetection
import requests 

# Constants
THRESHOLD = 0.95
MAX_ANNotATIONS = 40
IOU_THRESHOLD = 0.05

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    return model, image_processor, device

def model_results(img, model, image_processor, device):
    inputs = image_processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = image_processor.post_process_object_detection(outputs, THRESHOLD, target_sizes=torch.tensor([img.size[::-1]]).to(device))[0]

    person_scores = []
    person_labels = []
    person_boxes = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == 'person':
            person_labels.append(label)
            person_boxes.append(box)
            person_scores.append(score.item())  

    # Sorting the results by score
    sorted_indices = np.argsort(person_scores)[::-1]  

    sorted_person_labels = [person_labels[i] for i in sorted_indices]
    sorted_person_boxes = [person_boxes[i] for i in sorted_indices]
    sorted_person_scores = [person_scores[i] for i in sorted_indices]


    sorted_results = {
        "scores": sorted_person_scores,
        "labels": sorted_person_labels,
        "boxes": sorted_person_boxes
    }

    bboxes = [[int(x) for x in bbox] for bbox in sorted_results["boxes"]]
    
    return bboxes[:MAX_ANNotATIONS]

def get_iou(bbox1, box2):
    """
        Compute IoU of two bounding boxes.
        """
    x_left = max(bbox1[0], box2[0])
    y_top = max(bbox1[1], box2[1])
    x_right = min(bbox1[2], box2[2])
    y_bottom = min(bbox1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = bbox1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def remove_duplicates(bboxes):
    """Remove duplicate bounding boxes."""
    liste_tuple = [tuple(item) for item in bboxes]
    liste_sans_duplicatas = [list(item) for item in set(liste_tuple)]
    return liste_sans_duplicatas

def merge_overlapping_bboxes(bboxes, iou_threshold=0.3):
    merged_bboxes = []
    while bboxes:
        base = bboxes.pop(0)
        to_merge = [base]

        # Iteratively merge boxes
        for box in bboxes:
            if get_iou(base, box) > iou_threshold:
                to_merge.append(box)

        # Only keep boxes that were not merged
        bboxes = [box for box in bboxes if box not in to_merge]

        # Calculate the bounding box that encompasses all merged boxes
        merged_box = [
            min(box[0] for box in to_merge),
            min(box[1] for box in to_merge),
            max(box[2] for box in to_merge),
            max(box[3] for box in to_merge),
        ]
        merged_bboxes.append(merged_box)

    return merged_bboxes

def get_iou_annotations(anno_bboxes, model_bboxes):
    filtered_newbboxes = []
    newest_anno = []
    for newbbox in model_bboxes:
        overlaps = False
        for oldbbox in anno_bboxes:
            if get_iou(newbbox, oldbbox) > IOU_THRESHOLD:
                overlaps = True
                break
        if not overlaps and newbbox not in filtered_newbboxes:
            filtered_newbboxes.append(newbbox)
    fin = anno_bboxes + filtered_newbboxes
    newest_anno = merge_overlapping_bboxes(fin, iou_threshold=0.3)
    return newest_anno


def eliminer_duplicatas(liste):
    liste_tuple = [tuple(item) for item in liste]
    liste_sans_duplicatas = [list(item) for item in set(liste_tuple)]
    return liste_sans_duplicatas

def process_images(train_img, original_path, model, image_processor, device):
    list_appair = []
    for k, image in enumerate(train_img):
        if k % 100 == 0:
            print(f"Processed {k} images")
        img_path = os.path.join(original_path, image['folder'], image['file_name'])
        img = Image.open(img_path).convert("RGB")
        bboxes = model_results(img, model, image_processor, device)
        list_appair.append({'id': image['id'], 'bboxes': bboxes})
    return list_appair

def process_annotations(train_anno, list_appair):
    new_annotations = []
    new_id = 0
    for i, anno in enumerate(train_anno):
        img_id = anno['image_id']
        anno_bboxes = [anno['bbox']] if not isinstance(anno['bbox'][0], list) else anno['bbox']
        matched_appair = next((app for app in list_appair if app['id'] == img_id), None)
        if matched_appair:
            adjusted_bboxes = get_iou_annotations(anno_bboxes, matched_appair['bboxes'])
            adjusted_bboxes = get_iou_annotations(adjusted_bboxes,adjusted_bboxes)
            adjusted_bboxes = eliminer_duplicatas(adjusted_bboxes)
            adjusted_bboxes = merge_overlapping_bboxes(adjusted_bboxes, iou_threshold=0.2)
            for bbox in adjusted_bboxes:
                new_annotations.append({
                    'image_id': img_id,
                    'id': new_id,
                    'category_id': anno['category_id'],
                    'bbox': bbox,
                    'coco_ids': anno.get('coco_ids'),
                    'annotations_categories': anno.get('annotations_categories'),
                    'annotations_continuous': anno.get('annotations_continuous'),
                    'gender': anno.get('gender'),
                    'age': anno.get('age'),
                })
                new_id += 1
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} annotations")
    return new_annotations

def main():
    path = './new_annotations/EMOTIC_train_x1y1x2y2.json'
    #path = './test11095.json'
    with open(path, 'r') as file:
        train = json.load(file)

    original_path = "EMOTIC (1)/EMOTIC/PAMI/emotic"
    model, image_processor, device = load_model()

    list_appair = process_images(train['images'], original_path, model, image_processor, device)
    new_annotations = process_annotations(train['annotations'], list_appair)

    filename = './newest_' + os.path.basename(path)
    mixed_data = {'images': train['images'], 'annotations': new_annotations, 'categories': train['categories']}
    with open(filename, 'w') as f:
        json.dump(mixed_data, f)

    print("Processing completed.")

if __name__ == "__main__":
    main()