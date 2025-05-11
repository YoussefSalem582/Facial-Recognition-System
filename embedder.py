import os
import shutil
import random
import cv2
import numpy as np
from sklearn.metrics import classification_report
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib


# Setup InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Paths
original_faces_dir = "faces"
train_dir = "faces_train"
test_dir = "faces_test"

# Helpers
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"⚠️ No face detected in: {img_path}")
    return faces[0].embedding if faces else None


# Step 1: Filter out folders with <= 1 image
valid_people = []
for person in os.listdir(original_faces_dir):
    person_path = os.path.join(original_faces_dir, person)
    if not os.path.isdir(person_path):
        continue
    images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) > 1:
        valid_people.append((person, images))
    else:
        shutil.rmtree(person_path)  # Delete single-image person

# Step 2: Split train/test
for base_dir in [train_dir, test_dir]:
    ensure_dir(base_dir)

for person, images in valid_people:
    random.shuffle(images)
    split = int(len(images) * 0.7)
    train_images = images[:split]
    test_images = images[split:]

    for split_name, image_list in [("train", train_images), ("test", test_images)]:
        target_dir = os.path.join(train_dir if split_name == "train" else test_dir, person)
        ensure_dir(target_dir)
        for img in image_list:
            src = os.path.join(original_faces_dir, person, img)
            dst = os.path.join(target_dir, img)
            shutil.copy(src, dst)

# Step 3: Build embeddings DB from training set
face_db = {}

def get_valid_embedding(path):
    emb = get_embedding(path)
    return emb if emb is not None else None

for person in os.listdir(train_dir):
    person_path = os.path.join(train_dir, person)
    img_paths = [
        os.path.join(person_path, img)
        for img in os.listdir(person_path)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    embs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_valid_embedding, path): path for path in img_paths}
        for future in as_completed(futures):
            emb = future.result()
            if emb is not None:
                embs.append(emb)

    if embs:
        face_db[person] = np.mean(embs, axis=0)
    else:
        print(f"🚫 Skipping {person} — no valid embeddings found")

# Step 4: Predict test set
y_true = []
y_pred = []

def predict_image(person, img_path):
    emb = get_embedding(img_path)
    if emb is None:
        return None
    best_name = "Unknown"
    best_score = 0
    for ref_name, ref_emb in face_db.items():
        score = cosine_similarity(emb, ref_emb)
        if score > best_score:
            best_score = score
            best_name = ref_name
    prediction = best_name if best_score > 0.7 else "Unknown"
    return (person, prediction)

tasks = []
for person in os.listdir(test_dir):
    person_path = os.path.join(test_dir, person)
    for img in os.listdir(person_path):
        if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(person_path, img)
        tasks.append((person, img_path))

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(predict_image, person, path): (person, path) for person, path in tasks}
    for future in as_completed(futures):
        result = future.result()
        if result:
            yt, yp = result
            y_true.append(yt)
            y_pred.append(yp)

# Save
joblib.dump(face_db, "face_db.joblib")

# Step 5: Metrics
print("-> Classification Report (classes with \u22652 test images):")
test_counts = Counter(y_true)
filtered_y_true, filtered_y_pred = zip(*[
    (yt, yp) for yt, yp in zip(y_true, y_pred) if test_counts[yt] >= 2
])
print(classification_report(filtered_y_true, filtered_y_pred, zero_division=0))