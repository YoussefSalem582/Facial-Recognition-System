# recognizer.py
import os
import joblib
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Setup InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recognize_faces_on_image(image_path, output_path, face_db, threshold=0.7):
    img = cv2.imread(image_path)
    faces = app.get(img)
    for face in faces:
        emb = face.embedding
        name = "Unknown"
        best_score = 0.0
        for db_name, db_emb in face_db.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                name = db_name
        if best_score >= threshold:
            x1, y1, x2, y2 = map(int, face.bbox)
            box_thickness = max(2, int((y2 - y1) * 0.01))
            font_scale = max(0.6, (y2 - y1) / 100.0)
            font_thickness = 3
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            cv2.putText(img, name, (x1, y1 - 55), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 255, 0), font_thickness)
            cv2.putText(img, f"{best_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.85, (0, 255, 0), font_thickness)

    cv2.imwrite(output_path, img)

def recognize_faces_on_frame(frame, face_db, threshold=0.7):
    faces = app.get(frame)
    for face in faces:
        emb = face.embedding
        name = "Unknown"
        best_score = 0.0
        for db_name, db_emb in face_db.items():
            score = cosine_similarity(emb, db_emb)
            if score > best_score:
                best_score = score
                name = db_name
        if best_score >= threshold:
            x1, y1, x2, y2 = map(int, face.bbox)
            box_thickness = 2
            font_scale = max(0.6, (y2 - y1) / 200.0)
            font_thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
            cv2.putText(frame, name, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 255, 0), font_thickness)
            cv2.putText(frame, f"{best_score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.65, (0, 255, 0), font_thickness)
    return frame
