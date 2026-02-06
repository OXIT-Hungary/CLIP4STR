
# https://chatgpt.com/share/6985e963-4aac-800d-9f0e-fe35ce651790

import json
import os
from pathlib import Path
import cv2
from tqdm import tqdm

# =========================
# CONFIG
# =========================

COCO_JSON = "/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/tools/training_dataset/annotations/instances_Train.json"
IMAGES_ROOT = Path("/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/tools/training_dataset/images")

OUTPUT_DIR = Path("/home/chris/Documents/PROJECTS/CLIP4STR/code/CLIP4STR/tools/training_dataset/output")
OUTPUT_IMAGES = OUTPUT_DIR / "images"
LABEL_FILE = OUTPUT_DIR / "labels.txt"

BBOX_PADDING = 0.10   # optional extra margin around head

# =========================


def load_coco(path):
    with open(path, "r") as f:
        return json.load(f)


def build_image_lookup(coco):
    return {img["id"]: img for img in coco["images"]}


def resolve_image_path(file_name):
    """
    Robust resolver for nested folder structures.
    """
    p = Path(file_name)

    candidate = IMAGES_ROOT / p
    if candidate.exists():
        return candidate

    # fallback recursive search (slow but safe)
    matches = list(IMAGES_ROOT.rglob(p.name))
    if matches:
        return matches[0]

    return None


def crop_bbox(img, bbox):

    x, y, w, h = bbox

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    # padding
    pad_x = int((x2 - x1) * BBOX_PADDING)
    pad_y = int((y2 - y1) * BBOX_PADDING)

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    # clamp to image size
    h_img, w_img = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    return img[y1:y2, x1:x2]


def main():

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    coco = load_coco(COCO_JSON)
    image_lookup = build_image_lookup(coco)

    label_lines = []
    crop_index = 0

    for ann in tqdm(coco["annotations"]):

        # ---- Filter only visible numbers ----
        if not ann.get('attributes').get("NumberVisible").strip().lower() == "true":
            continue

        number = ann.get('attributes').get("PlayerNumber")
        if number is None:
            continue

        image_info = image_lookup.get(ann["image_id"])
        if image_info is None:
            continue

        image_path = resolve_image_path(image_info["file_name"])
        if image_path is None:
            print("Image not found:", image_info["file_name"])
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print("Failed to load:", image_path)
            continue

        crop = crop_bbox(img, ann["bbox"])

        if crop.size == 0:
            continue

        crop_index += 1
        filename = f"{crop_index:06d}.png"
        out_path = OUTPUT_IMAGES / filename

        cv2.imwrite(str(out_path), crop)

        label_lines.append(f"{filename} {number}")

    with open(LABEL_FILE, "w") as f:
        f.write("\n".join(label_lines))

    print("Done. Total crops:", crop_index)


if __name__ == "__main__":
    main()
