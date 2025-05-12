# Facial Recognition System

[![Visits](https://img.shields.io/badge/dynamic/json?color=blue&label=visits&query=value&url=https://api.countapi.xyz/hit/YoussefSalem582/Facial-Recognition-System)](https://github.com/YoussefSalem582/Facial-Recognition-System)

## 📋 Overview

This project implements an advanced facial recognition system using deep learning techniques for authentication and identification via images/video streams. Designed for security applications and access control systems, this solution leverages state-of-the-art computer vision algorithms and neural network architectures.

The system utilizes transfer learning with models like FaceNet and VGG-Face, optimized for low False Acceptance Rate (FAR) and real-time performance in various environmental conditions.

## ✨ Key Features

- **Advanced Face Detection**: Robust detection across various lighting conditions, angles, and occlusions
- **High-Accuracy Recognition**: Leveraging transfer learning with pre-trained models (FaceNet/VGG-Face)
- **Real-time Processing**: Optimized for low-latency applications with efficient inference
- **Low Error Rates**: Specifically engineered for minimal FAR (False Acceptance Rate) in security applications
- **Multi-face Processing**: Can handle multiple faces in frame simultaneously
- **Dataset Preparation**: Comprehensive processing of LFW and VGGFace datasets with face detection and augmentation
- **MLOps Implementation**: CI/CD pipeline for model training, evaluation, and deployment
- **Scalable Architecture**: Designed to handle growing datasets and user bases

## 🧠 Technical Implementation

### Architecture

The system follows a three-tier architecture:
1. **Face Detection Layer**: Uses a combination of HOG and CNN-based detectors to locate faces in images/video
2. **Feature Extraction Layer**: Employs pre-trained deep learning models to extract facial embeddings
3. **Classification Layer**: Uses similarity metrics and threshold-based classification to identify individuals

### Models & Techniques

- **Detection Models**: MTCNN / Haar Cascades with optimized parameters
- **Feature Extraction**: FaceNet (Inception-ResNet-v1) / VGG-Face (VGG-16 variation)
- **Distance Metrics**: Cosine similarity, Euclidean distance
- **Data Augmentation**: Rotation, scaling, lighting variation, horizontal flipping
- **Optimization**: Model quantization for deployment efficiency

## 🔧 Project Structure

```
facial-recognition-system/
├── facial_recognition_app/   # Main application code
│   ├── detector/            # Face detection implementation
│   ├── recognition/         # Recognition algorithms
│   ├── utils/               # Utility functions
│   └── app.py               # Main application entry point
├── Milestone 1/             # Milestone 1 Folder
│   ├── Facial Recognition System Mialiston One.pdf               # Report of Milestone one
│   └── Milestone_1_Data_Collection,_Exploration,_and_Preprocessing.ipynb               # Milestone 1 Notebook .ipynb file
├── faces_train/             # Training dataset
├── faces_test/              # Testing dataset
├── faces/                   # Sample face images
├── models/                  # Pretrained and fine-tuned models
├── embedder.py              # Face embedding generation
├── handler.py               # Request handling logic
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 99.38% |
| False Acceptance Rate (FAR) | 0.1% |
| False Rejection Rate (FRR) | 1.2% |
| F1 Score | 0.992 |
| Processing Time | ~50ms per frame |

## 🔬 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for real-time performance)
- OpenCV and related dependencies

### Setup
1. Clone the repository:
```bash
git clone https://github.com/YoussefSalem582/Facial-Recognition-System.git
cd Facial-Recognition-System
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download pre-trained models (if not included in repo):
```bash
python scripts/download_models.py
```

## 💻 Usage

### Basic Usage
```python
from facial_recognition_app import FacialRecognitionSystem

# Initialize the system
frs = FacialRecognitionSystem(model_type="facenet", threshold=0.6)

# Register a new face
frs.register_face("path/to/image.jpg", "person_name")

# Recognize faces in an image
results = frs.recognize("path/to/test_image.jpg")
for person, confidence in results:
    print(f"Detected: {person} with confidence {confidence:.2f}")
```

### Video Stream Processing
```python
import cv2
from facial_recognition_app import FacialRecognitionSystem

# Initialize the system
frs = FacialRecognitionSystem(model_type="facenet", threshold=0.6)

# Load registered faces
frs.load_registered_faces("path/to/faces_db")

# Process video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed_frame, results = frs.process_frame(frame)
    
    # Display results
    cv2.imshow("Facial Recognition", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🔄 Training

To train the system on your own dataset:

```bash
python scripts/train.py --dataset path/to/dataset --model facenet --epochs 50
```

## 🧪 Evaluation

Evaluate the system on a test dataset:

```bash
python scripts/evaluate.py --test-dir path/to/test_data --model models/facenet_custom.h5
```

## 🛣️ Roadmap

- [ ] Multi-factor authentication integration
- [ ] Emotion and age detection
- [ ] Support for masked face recognition
- [ ] Mobile application development
- [ ] REST API for cloud deployment
- [ ] Adversarial attack detection and prevention

## 👥 Contributors

This project was developed by:

- Ahmed Elrashidy - [@AhmedElrashidy11](https://github.com/AhmedElrashidy11)
- Ahmed Elsisi - [@ahmed-elsisi](https://github.com/ahmed-elsisi)
- Mostafa Eleimy - [@MostafaEleimy](https://github.com/MostafaEleimy)
- Youssef Ayman
- Youssef Salem - [@YoussefSalem582](https://github.com/YoussefSalem582)

## 🤝 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the existing coding style.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

- Ahmed Elrashidy - [@AhmedElrashidy11](https://github.com/AhmedElrashidy11)
- Ahmed Elsisi - [@ahmed-elsisi](https://github.com/ahmed-elsisi)
- Mostafa Eleimy - [@MostafaEleimy](https://github.com/MostafaEleimy)
- Youssef Ayman
- Youssef Salem - [@YoussefSalem582](https://github.com/YoussefSalem582)

Project Link: [https://github.com/YoussefSalem582/Facial-Recognition-System](https://github.com/YoussefSalem582/Facial-Recognition-System)

## 🙏 Acknowledgements

- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [VGGFace: Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
- [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/) 
