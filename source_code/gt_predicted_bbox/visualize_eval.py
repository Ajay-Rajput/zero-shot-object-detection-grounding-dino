import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ==== Inputs ====
image_folder = r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\dataset\one"
gt_json_path = r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\dataset\coco_annotations.json"
pred_json_path = r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\outputs\predictions_for_eval.json"
output_folder = r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\source_code\gt_predicted_bbox\visualized"

os.makedirs(output_folder, exist_ok=True)

# ==== Load JSONs ====
with open(gt_json_path, "r") as f:
    gt_data = json.load(f)

with open(pred_json_path, "r") as f:
    pred_data = json.load(f)

# Create mapping: image_id → filename
image_id_to_filename = {img["id"]: img["file_name"] for img in gt_data["images"]}

# Organize GT and predictions by image_id
gt_boxes_by_image = {}
for ann in gt_data["annotations"]:
    gt_boxes_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

pred_boxes_by_image = {}
for ann in pred_data:
    pred_boxes_by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

# ==== Draw boxes ====
for image_id, filename in image_id_to_filename.items():
    image_path = os.path.join(image_folder, filename)
    if not os.path.exists(image_path):
        print(f"Image {filename} not found, skipping...")
        continue

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw GT boxes in green
    for bbox in gt_boxes_by_image.get(image_id, []):
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    # Draw predicted boxes in red
    for bbox in pred_boxes_by_image.get(image_id, []):
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="blue", width=4)

    # Save visualized image
    output_path = os.path.join(output_folder, filename)
    img.save(output_path)
    print(f"Saved visualization: {output_path}")
