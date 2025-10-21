from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def evaluate_predictions(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_json", type=str, required=True, help="annotations.json file")
    parser.add_argument("--gt_json", type=str, required=True, help=r"C:\Users\ADMIN\Desktop\capston_project\GroundingDINO\dataset\coco_annotations.json")

    args = parser.parse_args()

    evaluate_predictions(args.gt_json, args.pred_json)
