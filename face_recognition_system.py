import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras_facenet import FaceNet

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = MTCNN()
        
        # FaceNet model (pre-trained)
        self.facenet_model = None
        self.load_facenet_model()
        
        # Classifier
        self.svm_model = SVC(kernel='linear', probability=True)
        self.label_encoder = LabelEncoder()
        
    def load_facenet_model(self):
        """FaceNet modelini yuklash"""
        # FaceNet modelini yuklash (keras-facenet orqali)
        self.facenet_model = FaceNet()
        print("FaceNet model yuklandi")
        
    
    def preprocess_face(self, img, box):
        """Yuzni preprocessing qilish"""
        x, y, w, h = box
        face = img[y:y+h, x:x+w]
        # 160x160 ga o'zgartirish (FaceNet uchun)
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32')
        # Normalizatsiya
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        return face
    
    def detect_faces(self, img):
        """Rasmdagi yuzlarni aniqlash"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_img)
        
        faces = []
        boxes = []

        for result in results:
            if result['confidence'] > 0.9:  # Ishonch darajasi
                box = result['box']
                face = self.preprocess_face(rgb_img, box)
                faces.append(face)
                boxes.append(box) 
        return faces, boxes
    
    def get_face_embedding(self, face):
        """Yuz uchun embedding olish"""
        if self.facenet_model is None:
            return None
            
        face = np.expand_dims(face, axis=0)
        embedding = self.facenet_model.embeddings(face)
        return embedding[0]
    
    def load_dataset(self, dataset_path):
        """Dataset yuklash va embeddings olish"""
        embeddings = []
        labels = []
        
        print("Dataset yuklanmoqda...")
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            print(f"Processing: {person_name}")
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    faces, _ = self.detect_faces(img)
                    
                    for face in faces:
                        embedding = self.get_face_embedding(face)
                        if embedding is not None:
                            embeddings.append(embedding)
                            labels.append(person_name)
                            
                except Exception as e:
                    print(f"Xatolik: {img_path} - {e}")
        
        return np.array(embeddings), np.array(labels)
    
    def train_classifier(self, dataset_path):
        """Klassifikatorni o'rgatish"""
        print("Klassifikator o'rgatilmoqda...")
        
        # Dataset yuklash
        embeddings, labels = self.load_dataset(dataset_path)
        
        if len(embeddings) == 0:
            print("Dataset bo'sh!")
            return False
        # Label encoding
        encoded_labels = self.label_encoder.fit_transform(labels)
        # SVM o'rgatish
        self.svm_model.fit(embeddings, encoded_labels)
        # Modelni saqlash
        self.save_model()
        
        print(f"O'rgatish tugadi. {len(set(labels))} kishi, {len(embeddings)} rasm")
        return True
    
    def save_model(self):
        """Modelni saqlash"""
        model_data = {
            'svm_model': self.svm_model,
            'label_encoder': self.label_encoder
        }
        
        with open('face_recognition_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saqlandi: face_recognition_model.pkl")
    def load_model(self):
        """Modelni yuklash"""
        try:
            with open('face_recognition_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            self.svm_model = model_data['svm_model']
            self.label_encoder = model_data['label_encoder']
            print("Model yuklandi")
            return True
        except:
            print("Model topilmadi")
            return False
    
    def predict_face(self, img):
        """Yuzni bashorat qilish"""
        faces, boxes = self.detect_faces(img)
        
        results = []
        
        for i, face in enumerate(faces):
            embedding = self.get_face_embedding(face)
            
            if embedding is not None:
                # Bashorat
                prediction = self.svm_model.predict([embedding])[0]
                confidence = self.svm_model.predict_proba([embedding]).max()
                
                # Label decode
                person_name = self.label_encoder.inverse_transform([prediction])[0]
                
                results.append({
                    'box': boxes[i],
                    'name': person_name,
                    'confidence': confidence
                })
        
        return results
    
    def test_model(self, test_dataset_path):
        """Modelni test qilish"""
        print("Model test qilinmoqda...")
        
        true_labels = []
        predicted_labels = []
        accuracy_history = {}
        for person_name in os.listdir(test_dataset_path):
            person_path = os.path.join(test_dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    results = self.predict_face(img)
                    
                    if results:
                        predicted_name = results[0]['name']
                        if predicted_name != person_name:
                            print(f"Xato: {person_name} - Bashorat: {predicted_name}")
                            print(f"Rasm: {img_path}", " \nIshonch:", results[0]['confidence'], '\n')

                        true_labels.append(person_name)
                        predicted_labels.append(predicted_name)
                    
                except Exception as e:
                    continue
            if len(true_labels) > 0:
                accuracy = accuracy_score(true_labels, predicted_labels)
                print(f"Test aniqligi: {accuracy:.2f}")
                accuracy_history[person_name] = accuracy
            else:
                print("Test ma'lumotlari topilmadi")
                accuracy_history[person_name] = 0
        return accuracy_history

# Foydalanish namunasi
def main():
    # Face Recognition tizimini yaratish
    fr_system = FaceRecognitionSystem()
    
    # 1. Modelni o'rgatish
    # dataset_path = "facerec_datasets/Dataset/Faces"
    # if os.path.exists(dataset_path):
    #     fr_system.train_classifier(dataset_path)
    
    # 2. Modelni yuklash (agar avval o'rgatilgan bo'lsa)
    fr_system.load_model()
    
    # 3. Test qilish
    # test_path = "person_dataset/validation"
    # if os.path.exists(test_path):
    #     accuracy_history = fr_system.test_model(test_path)
    #     print("Test aniqligi tarixi:", accuracy_history)
    
    image_path = "tests/alexandra-daddario-2022-noms-450x600.jpg"
    # image prediction
    predicted_person = fr_system.predict_face(cv2.imread(image_path))
    if predicted_person:
        print(f"Bashorat: {predicted_person[0]['name']}, Ehtimollik: {predicted_person[0]['confidence']:.4f}")
    else:
        print("Yuz aniqlanmadi yoki bashorat qilinmadi")

    # 4. Real-time test
    # cap = cv2.VideoCapture(2)
    
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    #     # Face recognition
    #     results = fr_system.predict_face(frame)
        
    #     # Natijalarni chizish
    #     for result in results:
    #         box = result['box']
    #         name = result['name']
    #         confidence = result['confidence']
            
    #         x, y, w, h = box
            
    #         # To'rtburchak chizish
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    #         # Nom va ishonch darajasini yozish
    #         text = f"{name} ({confidence:.2f})"
    #         cv2.putText(frame, text, (x, y-10), 
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    #     cv2.imshow('Face Recognition', frame)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
