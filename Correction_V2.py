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
MAX_ANNotATIONS = 30

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

def iou(box1, box2):
    """Calculates Intersection Over Union (IOU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    interArea = max(0, min(x2, x2g) - max(x1, x1g)) * max(0, min(y2, y2g) - max(y1, y1g))
    boxAArea = (x2 - x1) * (y2 - y1)
    boxBArea = (x2g - x1g) * (y2g - y1g)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

"""
def remove_duplicates(lst):
    return [list(t) for t in {tuple(item) for item in lst}]
"""

def remove_duplicates(bboxes):
    """Remove duplicate bounding boxes."""
    unique_bboxes = []
    [unique_bboxes.append(x) for x in bboxes if x not in unique_bboxes]
    return unique_bboxes

def get_iou_annotations(anno_bboxes, model_bboxes):
    # Copie des boîtes d'annotations originales
    new_annots = anno_bboxes.copy()
    added_bboxes = []  # Liste pour les boîtes prédites à ajouter

    for anno_bbox in anno_bboxes:
        for model_bbox in model_bboxes:
            if iou(anno_bbox, model_bbox) >= 0.5:
                # treshold à augmenter ou garder un bbox sur 2
                added_bboxes.append(model_bbox)
                break  # Garde le 'break' si une seule correspondance est nécessaire

    # Ajoute les nouvelles boîtes tout en évitant les doublons potentiels
    for bbox in added_bboxes:
        if bbox not in new_annots:
            new_annots.append(bbox)

    # Assumant que remove_duplicates est bien implémentée pour des boîtes englobantes
    new_annots = remove_duplicates(new_annots)
    return new_annots

"""

def get_iou_annotations(anno_bboxes, model_bboxes,threshold=0.0001):
    # Adjusts annotation bounding boxes based on IOU with model predictions
    new_annots = []
    keep_flags = [True] * len(model_bboxes)
    for i, model_bbox in enumerate(model_bboxes):
        for anno_bbox in anno_bboxes: 
            if iou(anno_bbox, model_bbox) >= threshold:
                keep_flags[i] = False
                break 

    adjusted_model_bboxes = [bbox for bbox, keep in zip(model_bboxes, keep_flags) if keep]

    new_annots = remove_duplicates(anno_bboxes + adjusted_model_bboxes)

    return new_annots
"""

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