# Convert the Large-Scale Fish Dataset (image + GT mask pairs) into YOLOv8-seg polygon labels.
# Run from the repository root:  python scripts/make_yolo_seg.py
import os, cv2, glob, random, shutil
import numpy as np

# ---- configure these ----
ROOT = r"data/LARGE SCALE FISH DATASET/Fish_Dataset/Fish_Dataset"
OUT  = r"data/YOLO_FISH_SEG"  # matches configs/fish_seg.yaml
VAL_FRACTION = 0.15
RANDOM_SEED = 42
MIN_POLY_POINTS = 12      # increase for smoother polygons
APPROX_EPS_FRAC = 0.002   # polygon simplification; lower = more points
# -------------------------

random.seed(RANDOM_SEED)
os.makedirs(os.path.join(OUT, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "images", "val"),   exist_ok=True)
os.makedirs(os.path.join(OUT, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUT, "labels", "val"),   exist_ok=True)

# collect all species subfolders (each has "<Species>\ <Species> GT")
species_dirs = [d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)]

pairs = []  # (img_path, mask_path)
for sp_dir in species_dirs:
    # find image folder and GT folder
    subs = [d for d in glob.glob(os.path.join(sp_dir, "*")) if os.path.isdir(d)]
    img_dir = [d for d in subs if os.path.basename(d).lower().endswith("gt")==False]
    gt_dir  = [d for d in subs if os.path.basename(d).lower().endswith("gt")==True]
    if not img_dir or not gt_dir:
        continue
    img_dir, gt_dir = img_dir[0], gt_dir[0]

    # index masks by basename (without extension)
    mask_index = {}
    for m in glob.glob(os.path.join(gt_dir, "*")):
        base = os.path.splitext(os.path.basename(m))[0]
        mask_index[base] = m

    # match images to masks
    for img in glob.glob(os.path.join(img_dir, "*")):
        base = os.path.splitext(os.path.basename(img))[0]
        # handle occasional naming differences (spaces, case)
        cand = [
            base,
            base.replace(" ", "_"),
            base.replace("_", " "),
            base.lower(), base.upper()
        ]
        mask = None
        for c in cand:
            if c in mask_index:
                mask = mask_index[c]
                break
        if mask is None:
            # fallback: try any mask that startswith the same prefix
            for k,v in mask_index.items():
                if k.startswith(base):
                    mask = v; break
        if mask:
            pairs.append((img, mask))

# split
random.shuffle(pairs)
n_val = int(len(pairs) * VAL_FRACTION)
val_pairs = set(pairs[:n_val])

def write_label(mask_path, out_txt):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return False
    # binarize
    _, bw = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    # find contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    # take largest contour (assume single fish)
    cnt = max(contours, key=cv2.contourArea)

    # approximate polygon
    eps = APPROX_EPS_FRAC * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

    # ensure minimum points (fallback to raw contour if too few)
    if len(approx) < MIN_POLY_POINTS:
        approx = cnt.reshape(-1, 2)

    h, w = bw.shape[:2]
    # normalize to [0,1]
    poly = []
    for x,y in approx:
        poly += [x / w, y / h]

    # YOLOv8-seg: class_id followed by x y pairs
    with open(out_txt, "w", encoding="utf-8") as f:
        line = "0 " + " ".join(f"{v:.6f}" for v in poly)
        f.write(line + "\n")
    return True

# convert & copy
for img, mask in pairs:
    split = "val" if (img, mask) in val_pairs else "train"
    out_img = os.path.join(OUT, "images", split, os.path.basename(img))
    out_txt = os.path.join(OUT, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    ok = write_label(mask, out_txt)
    if ok:
        shutil.copy2(img, out_img)

print("Done. Created YOLO dataset at:", OUT)
