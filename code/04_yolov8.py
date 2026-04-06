import os, shutil
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from config import BASE_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR

TARGET = os.path.join(BASE_DIR, 'car_damage_yolo')
CLASS_MAPPING = {'01-minor': 0, '02-moderate': 1, '03-severe': 2}

# ── Build YOLO folder structure (train, val, test) ────────────────────────────
for split, src_dir in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
    os.makedirs(f'{TARGET}/images/{split}', exist_ok=True)
    os.makedirs(f'{TARGET}/labels/{split}', exist_ok=True)

    for class_folder, class_id in CLASS_MAPPING.items():
        img_dir   = os.path.join(src_dir, class_folder)
        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in tqdm(img_files, desc=f'{split} — {class_folder}'):
            src = os.path.join(img_dir, img_file)
            dst = os.path.join(TARGET, 'images', split, img_file)
            lbl = os.path.join(TARGET, 'labels', split,
                               img_file.rsplit('.', 1)[0] + '.txt')

            shutil.copyfile(src, dst)
            with Image.open(src) as im:
                w, h = im.size
            with open(lbl, 'w') as f:
                f.write(f'{class_id} 0.5 0.5 0.8 0.8\n')

# ── Write data.yaml ───────────────────────────────────────────────────────────
yaml_content = f"""
path: {TARGET}
train: images/train
val:   images/val
test:  images/test

names:
  0: minor
  1: moderate
  2: severe
"""
with open(f'{TARGET}/data.yaml', 'w') as f:
    f.write(yaml_content)

# ── Train ─────────────────────────────────────────────────────────────────────
model = YOLO('yolov8n.pt')
model.train(
    data=f'{TARGET}/data.yaml',
    epochs=30,          # practical for CPU
    imgsz=416,          # smaller image = faster on CPU
    batch=8,            # reduced for CPU
    name='car_damage_yolo',
    device='cpu'
)

# ── Validate ──────────────────────────────────────────────────────────────────
val_results = model.val(
    data=f'{TARGET}/data.yaml',
    split='val'
)
print("\n── Validation Results ──")
print(val_results)

# ── Test ──────────────────────────────────────────────────────────────────────
test_results = model.val(
    data=f'{TARGET}/data.yaml',
    split='test'
)
print("\n── Test Results ──")
print(test_results)

# ── Sample predictions ────────────────────────────────────────────────────────
sample_imgs = [
    os.path.join(TEST_DIR, '01-minor'),
    os.path.join(TEST_DIR, '02-moderate'),
    os.path.join(TEST_DIR, '03-severe'),
]

best_model = YOLO(f'runs/detect/car_damage_yolo/weights/best.pt')
for folder in sample_imgs:
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    for img_path in imgs:
        results = best_model.predict(source=img_path, conf=0.25, save=True)
        print(f"Predicted: {img_path} → {results[0].boxes}")