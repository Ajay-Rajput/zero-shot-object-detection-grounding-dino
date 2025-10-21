import os
import json
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(config_file, checkpoint_path):
    args = SLConfig.fromfile(config_file)
    args.device = "cpu"
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, token_spans=None):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f" ({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases


def plot_boxes_to_image(image_pil, boxes, labels):
    W, H = image_pil.size
    draw = ImageDraw.Draw(image_pil)

    for box, label in zip(boxes, labels):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = map(int, box.tolist())

        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

        font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((x0, y0), label, font=font)
        else:
            w, h = draw.textsize(label, font=font)
            bbox = (x0, y0, x0 + w, y0 + h)

        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)

    return image_pil


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.config_file, args.checkpoint_path)

    images = [f for f in os.listdir(args.image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    annotations = []
    predictions = []

    image_id = 1  # Start image ID from 1

    for img_filename in tqdm(images, desc="Processing images"):
        image_path = os.path.join(args.image_folder, img_filename)
        image_pil, image_tensor = load_image(image_path)

        boxes, phrases = get_grounding_output(
            model, image_tensor, args.text_prompt, args.box_threshold, args.text_threshold
        )

        if len(boxes) == 0:
            print(f"No boxes found in image: {img_filename}")
            image_id += 1
            continue

        # Save visualization annotation
        annotation = {
            "image_filename": img_filename,
            "boxes": boxes.tolist(),
            "labels": phrases
        }
        annotations.append(annotation)

        # Save COCO-format prediction
        W, H = image_pil.size
        for box, phrase in zip(boxes, phrases):
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = map(float, box.tolist())
            w_abs = x1 - x0
            h_abs = y1 - y0

            score = float(phrase.split("(")[-1].rstrip(")"))

            pred_entry = {
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x0, y0, w_abs, h_abs],
                "score": score
            }
            predictions.append(pred_entry)

        # Save image with boxes drawn
        image_with_boxes = plot_boxes_to_image(image_pil, boxes, phrases)
        image_with_boxes.save(os.path.join(args.output_dir, img_filename))

        image_id += 1

    # Save outputs
    with open(os.path.join(args.output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=4)

    with open(os.path.join(args.output_dir, "predictions_for_eval.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Saved {len(annotations)} annotations to {args.output_dir}/annotations.json")
    print(f"Saved {len(predictions)} COCO predictions to {args.output_dir}/predictions_for_eval.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    args = parser.parse_args()

    main(args)
