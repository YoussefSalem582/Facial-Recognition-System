import cv2
import numpy as np
import joblib
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# --- Load face database ---
with open("face_db.joblib", "rb") as f:
    face_db = joblib.load("face_db.joblib")

# --- Setup InsightFace ---
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# --- Recognize faces in input image ---
def recognize_faces(image_path, output_path="output.jpg", threshold=0.7):
    img = cv2.imread(image_path)
    faces = app.get(img)

    for face in faces:
        emb = face.embedding
        name = "Unknown"
        best_score = 0

        for db_name, db_emb in face_db.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                name = db_name

        if best_score < threshold:
            name = "Unknown"

        # Draw bounding box + label
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} ({best_score:.2f})" if name != "Unknown" else name
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"✅ Saved output with labels to: {output_path}")

# --- Example usage ---
recognize_faces("test_photo.jpg")
