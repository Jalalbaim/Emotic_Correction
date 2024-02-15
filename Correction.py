# libraries
import torch
import numpy as np
import torchvision.transforms as T 
from PIL import Image
from torchvision.transforms import functional as F
import os
import json

"""
Description:

This script is written to correct the Emotic Dataset annotations using the DETR model as a SOA model for correction in order to improve our BeNeT Scores.

@author: Jalal
"""
# Check for CUDA and set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Importing model 
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(device)
model.eval()

# functions

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def model_results(img):
    img = transform(img).unsqueeze(0).to(device)
    outputs = model(img)
    # keep only predictions with 0.9+ confidence and labeled as "person"
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = (probas.max(-1).values > 0.9) & (probas.argmax(-1) == 1)  # Filter for "person" class
    # convert boxes from [0; 1] to image scales
    # Correcting the line causing TypeError
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size()[2:])
    bboxes_scaled = bboxes_scaled.tolist()
    return bboxes_scaled , probas[keep]

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    # determine the coordinates of the intersection rectangle
    xA = max(x1, x1g)
    yA = max(y1, y1g)
    xB = min(x2, x2g)
    yB = min(y2, y2g)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x2g - x1g + 1) * (y2g - y1g + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_iou(bbox1, bbox2, thresh, new_annots = []):
    iou_score = iou(bbox1, bbox2)
    if iou_score < thresh:
        new_annots.append(bbox1)
        new_annots.append(bbox2)
    else:
        pass
    return new_annots

def remove_duplicates(lst):
    return [list(t) for t in {tuple(item) for item in lst}]

# Main

def main():

    ## Loadin the JSON = {images: [{}], annotations: [{}]}

    path = './new_annotations/EMOTIC_train_x1y1x2y2.json'
    train = json.load(open(path))
    train_anno = train['annotations'] # dictionnary of annotations
    train_img = train['images'] # dictionnary of images

    ## Processing images part 

    original_path = "EMOTIC (1)/EMOTIC/PAMI/emotic"
    # liste d'appairement des images et des annotations en dictionnaires
    list_appair = []
    i = 0
    for image in train_img:
        # image est le dictionnaire d'information d'une image
        # train image est le dictionnaire d'information de toutes les images
        if i<100: # pour tester sur les 100 premières images
            file_name = image['file_name']
            folder = image['folder']
            img_path = original_path + '/' + folder + '/' + file_name
            img = Image.open(img_path)
            bboxes , probas = model_results(img)
            list_appair.append({'id': image['id'], 'bboxes': bboxes})
        i+=1
    
    ## Processing annotations part 
        
    train_new_annots = train_anno

    for i, anno in enumerate(train_new_annots):
        if i < 100: # pour tester sur les 100 premières images
            img_id = anno['image_id']
            anno_id = anno['id']
            bbox = anno['bbox']
            
            # Check if bbox is not already a list of lists
            if not isinstance(bbox[0], list):
                bbox = [bbox]
            
            #print(img_id, anno_id, bbox)
            
            for appair in list_appair:
                if appair['id'] == img_id:
                    # Initialize new annotations list for managing bboxes for each image
                    new_annots = []
                    
                    for single_bbox in bbox:  # Use single_bbox to avoid confusion with the outer bbox
                        for bbox2 in appair['bboxes']:
                            new_annots = get_iou(single_bbox, bbox2, 0.99, new_annots)
                    
                    anno['bbox'] = remove_duplicates(new_annots)
            # Extend the annotations categories list with None for the remaining bboxes
            if len(train_anno[i]['bbox']) > len(train_new_annots[i]['annotations_categories']):
                train_new_annots[i]['annotations_categories'].extend([None] * (len(train_anno[i]['bbox']) - len(train_new_annots[i]['annotations_categories'])))
    
    # Save the new annotations as a JSON file
                
    filename = './newest' + os.path.basename(path)

    # Create a dictionary with the images and annotations
    mixed_data = {'images': train_img, 'annotations': train_new_annots}

    # Save the mixed data as a JSON file
    with open(filename, 'w') as f:
        json.dump(mixed_data, f)



if __name__ == "__main__":
    main()