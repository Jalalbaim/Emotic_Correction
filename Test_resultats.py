## Test des resultats 
import torch
import numpy as np
import torchvision.transforms as T 
from PIL import Image
from torchvision.transforms import functional as F
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches


"""
This script will be run to test the results of the model on the images of the dataset.
@author: Jalal
"""

    
# Check for CUDA and set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Importing model 
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(device)
model.eval()


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.Lambda(lambda img: img.convert('RGB')),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def transform_image(img):
    transformed_img = transform(img)
    return transformed_img

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
    img = transform_image(img).unsqueeze(0).to(device)
    outputs = model(img)
    # keep only predictions with 0.9+ confidence and labeled as "person"
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = (probas.max(-1).values > 0.9) & (probas.argmax(-1) == 1)  # Filter for "person" class
    # convert boxes from [0; 1] to image scales
    # Correcting the line causing TypeError
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size()[2:])
    bboxes_scaled = bboxes_scaled.tolist()
    bboxes_scaled = bboxes_scaled[:25]
    return bboxes_scaled , probas[keep]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def main():
    
    path = "EMOTIC (1)\EMOTIC\PAMI\emotic\mscoco\images\COCO_val2014_000000482242.jpg"
    path = path.replace("\\", "/")  
    image = Image.open(path)
    """
    image = plt.imread(path)
    plt.imshow(image)
    """
    bboxes_scaled , probas = model_results(image)

    plot_results(image, probas, bboxes_scaled)

if __name__ == "__main__":
    main()