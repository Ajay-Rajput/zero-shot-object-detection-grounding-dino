import os
import json
from PIL import Image

# Paths
image_folder = r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\dataset\subset"         # Your image folder path
annotation_folder = r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\dataset\Open Images Dataset v6 Foods.v4i.yolov11\valid\labels"  # Your annotation TXT files
output_json = r'C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\dataset\coco_annotations.json'

# COCO structure with info
coco_output = {
    "info": {
        "description": "My custom dataset",
        "version": "1.0",
        "year": 2025,
        "contributor": "ajay",
        "date_created": "2025/09/14"
    },
    "images": [],
    "annotations": [],
    "categories": []
}

category_set = set()
annotation_id = 1
image_id = 1

# Iterate over images
for img_filename in os.listdir(image_folder):
    if not img_filename.lower().endswith(('.jpg', '.png')):
        continue

    image_path = os.path.join(image_folder, img_filename)
    txt_filename = os.path.splitext(img_filename)[0] + '.txt'
    txt_path = os.path.join(annotation_folder, txt_filename)

    # Load image size
    image = Image.open(image_path)
    width, height = image.size

    # Add image info
    coco_output['images'].append({
        "file_name": img_filename,
        "height": height,
        "width": width,
        "id": image_id
    })

    # Process annotation file
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip wrong formatted lines

            class_id, x_center, y_center, w, h = map(float, parts)
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            x_min = x_center - w / 2
            y_min = y_center - h / 2

            category_set.add(int(class_id))

            coco_output['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x_min, y_min, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    image_id += 1

# Add categories
for cat_id in sorted(category_set):
    coco_output['categories'].append({
        "id": cat_id,
        "name": str(cat_id),
        "supercategory": "none"
    })

# Save to JSON
with open(output_json, 'w') as json_file:
    json.dump(coco_output, json_file, indent=4)

print(f"COCO annotation saved to {output_json}")
