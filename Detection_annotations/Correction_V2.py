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
MAX_ANNotATIONS = 50
IOU_THRESHOLD = 0.9

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
    
    return bboxes

def get_iou(bbox1, bbox2):
    """Calcul de l'Intersection over Union (IoU) entre deux bboxes."""
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2
    
    # Calcul de l'aire d'intersection
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # Aire des bboxes
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    # Calcul de l'IoU
    union_area = bbox1_area + bbox2_area - inter_area
    IoU = inter_area / union_area
    return IoU

"""
def remove_duplicates(lst):
    return [list(t) for t in {tuple(item) for item in lst}]
"""

def remove_duplicates(bboxes):
    """Remove duplicate bounding boxes."""
    liste_tuple = [tuple(item) for item in bboxes]
    liste_sans_duplicatas = [list(item) for item in set(liste_tuple)]
    return liste_sans_duplicatas

"""
def get_iou_annotations(anno_bboxes: list, model_bboxes: list):
    filtered_newbboxes = []
    newest_anno = []
    for newbbox in model_bboxes:
        overlaps = False
        for oldbbox in anno_bboxes:
            iou = get_iou(newbbox,oldbbox)
            print("Iou",iou)
            if iou > IOU_THRESHOLD:
                overlaps = True
                break
        if not overlaps and newbbox not in filtered_newbboxes:
            filtered_newbboxes.append(newbbox)
    
    fein = anno_bboxes + filtered_newbboxes

    newest_anno = remove_duplicates(fein)
    return newest_anno[:MAX_ANNotATIONS]


def eliminer_duplicatas(liste):
    liste_tuple = [tuple(item) for item in liste]
    liste_sans_duplicatas = [list(item) for item in set(liste_tuple)]
    return liste_sans_duplicatas
"""
def get_iou_annotations(anno_bboxes, model_bboxes,threshold=0.0001):
     # Copie des boîtes d'annotations originales
    new_annots = anno_bboxes.copy()
    added_bboxes = []  # Liste pour les boîtes prédites à ajouter

    for anno_bbox in anno_bboxes:
        for model_bbox in model_bboxes:
            if get_iou(anno_bbox, model_bbox) >= IOU_THRESHOLD:
                # treshold à augmenter ou garder un bbox sur 2
                added_bboxes.append(model_bbox)
                break  

    # Ajoute les nouvelles boîtes tout en évitant les doublons potentiels
    for bbox in added_bboxes:
        if bbox not in new_annots:
            new_annots.append(bbox)

    # Assumant que remove_duplicates est bien implémentée pour des boîtes englobantes
    new_annots = remove_duplicates(new_annots)
    return new_annots

"""
def get_iou_annotations(anno_bboxes, model_bboxes):
    # Initialisation de la liste finale des annotations
    final_annotations = anno_bboxes.copy()

    # Parcourir chaque bbox modèle pour le comparer avec les bboxes existants
    for model_bbox in model_bboxes:
        max_iou = 0.0  # Initialiser le IoU maximum trouvé pour le bbox modèle actuel
        # Comparer avec chaque bbox d'annotation
        for anno_bbox in anno_bboxes:
            iou = get_iou(anno_bbox, model_bbox)
            max_iou = max(max_iou, iou)  # Mise à jour du IoU maximum

        # Si l'IoU maximum est inférieur au seuil, ajouter le bbox modèle à la liste finale
        if max_iou < IOU_THRESHOLD:
            final_annotations.append(model_bbox)
    
    final_annotations = remove_duplicates(final_annotations) 

    return final_annotations
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
    # Initial setup remains the same
    new_annotations = []
    new_id = 0
    annotations_by_image_id = {}

    for img in train_anno:
        annotation = {
            'bbox': img['bbox'],
            'category_id': img['category_id'],
            'annotations_categories': img.get('annotations_categories', []),
            'annotations_continuous': img.get('annotations_continuous', {}),
            'gender': img.get('gender', None),
            'age': img.get('age', None),
            'coco_ids': img.get('coco_ids', None),

        }

        if img["image_id"] not in annotations_by_image_id:
            annotations_by_image_id[img["image_id"]] = [annotation]
        else:
            annotations_by_image_id[img["image_id"]].append(annotation)
    i=0
    for img_id, annotations in annotations_by_image_id.items():
        anno_bboxes = [anno['bbox'] if not isinstance(anno['bbox'][0], list) else anno['bbox'] for anno in annotations]
        matched_appair = next((app for app in list_appair if app['id'] == img_id), None)
        if matched_appair:
            adjusted_bboxes = get_iou_annotations(anno_bboxes, matched_appair['bboxes'])
            # [[start_x, start_y, end_x, end_y], [start_x, start_y, end_x, end_y], ...]
            
            annotations_by_image_id[img_id]
            for i, anno in enumerate(adjusted_bboxes):
                #try:
                #annotations[i]['bbox'] = anno
                new_id += 1
                new_annotations.append({
                    'image_id': img_id,
                    'id': new_id,
                    'bbox': anno,
                    'category_id': annotations[i].get('category_id', 0) if i < len(annotations) else 0,
                    'annotations_categories': annotations[i].get('annotations_categories', []) if i < len(annotations) else [],
                    'annotations_continuous': annotations[i].get('annotations_continuous', {}) if i < len(annotations) else {},
                    'gender': annotations[i].get('gender', None) if i < len(annotations) else None,
                    'age': annotations[i].get('age', None) if i < len(annotations) else None,
                    'coco_ids': annotations[i].get('coco_ids', None) if i < len(annotations) else None,
                })

                """
                except:
                    print("image_id",img_id)
                    print("annotations",annotations)
                    continue
                """
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} annotations")
        #len(new_annotations)
    return new_annotations  

def main():
    path = './new_annotations/EMOTIC_test_x1y1x2y2.json'
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

# End of Correction_V2.py