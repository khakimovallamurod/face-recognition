from mtcnn.mtcnn import MTCNN
import cv2
import os
from matplotlib import pyplot as plt
from keras_facenet import FaceNet
import numpy as np

embedder = FaceNet()
detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face

def load_faces(directory):
    faces = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        face = extract_face(path)
        if face is not None:
            faces.append(face)
    return faces



def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = embedder.embeddings(face_pixels)
    return embedding[0]


def prepare_dataset(directory):
    X, y = [], []
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

X_faces, y_faces = prepare_dataset('augmented_faces')
X_embeddings = np.array([get_embedding(face) for face in X_faces])
np.savez_compressed('faces_embeddings.npz', X_embeddings, y_faces)

