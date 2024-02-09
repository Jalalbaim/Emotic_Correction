# Emotic_Correction

""" 
Description: This script is used to correct the Emotic dataset annotations.
Here we correct the annotations of the Emotic dataset in order to improve our BeNeT Scores.

@author: Jalal
"""

# Importing Libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T 
from PIL import Image
from torchvision.transforms import functional as F
import os
import json

# Importing model 
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
model.eval()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


emotic_folder = "EMOTIC (1)/EMOTIC/PAMI/emotic/framesdb/images/"

def save_outputs(output_folder, bbox, probas, image_path):
    output_list = []

    # Preprocess bbox and probas for JSON serialization
    bbox_list = bbox.tolist()  # Convert bbox tensor to a list of lists
    probas_list = probas.tolist()  # Convert probas tensor to a list

    # Correctly round the bbox coordinates
    rounded_bbox_list = [[round(coord, 2) for coord in bbox] for bbox in bbox_list]

    # Construct the output dictionary
    image_file = os.path.basename(image_path)  # Extract image file name from path
    img_name = os.path.splitext(image_file)[0]
    output_dict = {
        'name': img_name,
        'folder': output_folder,
        'bbox': rounded_bbox_list,
        'score': probas_list
    }

    output_list.append(output_dict)
    
    output_file = os.path.join(output_folder, 'output.json')
    with open(output_file, 'w') as f:
        json.dump(output_list, f, indent=4)  # Use indent for pretty printing
    
    print('Outputs saved to:', output_file)


# Get the list of image files in the folder
image_files = [file for file in os.listdir(emotic_folder) if file.endswith(".jpg") or file.endswith(".png")]

j = 0
for image_file in image_files:
    if j<10:
        # Construct the full path to the image file
        image_path = os.path.join(emotic_folder, image_file)
        
        # Load and display the image
        image = Image.open(image_path)
        img = transform(image).unsqueeze(0)
        outputs = model(img)
        # keep only predictions with 0.9+ confidence and labeled as "person"
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = (probas.max(-1).values > 0.9) & (probas.argmax(-1) == 1)  # Filter for "person" class
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
        # Save the outputs
        output_folder = "./outputs"
        save_outputs(output_folder, bboxes_scaled, probas[keep], image_path)
        j+=1

# Fin