# app.py
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import os
import joblib
from model.recognizer import recognize_faces_on_image, recognize_faces_on_frame, cosine_similarity

app = Flask(__name__)

# Load face DB
face_db = joblib.load("model/face_db.joblib")

# Camera feed (live mode)
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        annotated = recognize_faces_on_frame(frame, face_db)
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return render_template('index.html', error="No photo uploaded")

    photo = request.files['photo']
    if photo.filename == '':
        return render_template('index.html', error="No selected file")

    img_path = os.path.join("static", "uploaded.jpg")
    photo.save(img_path)

    output_path = os.path.join("static", "result.jpg")
    recognize_faces_on_image(img_path, output_path, face_db)

    return render_template('index.html', result_path='static/result.jpg')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
