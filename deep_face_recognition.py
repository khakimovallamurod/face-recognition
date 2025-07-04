import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
from deepface import DeepFace
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionSystem:
    def __init__(self):
        self.detector = MTCNN()
        self.model = None
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        
    def detect_and_extract_face(self, image_path, target_size=(224, 224)):
        """
        MTCNN yordamida yuzni aniqlash va ajratib olish
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = image.copy()
        cv2.imwrite("image_rgb001.png", image_rgb)
        print("Image RGB saved as image_rgb001.png")
        detections = self.detector.detect_faces(image_rgb)

        if len(detections) == 0:
            return None
        
        detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = detection['box']
        face = image_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        cv2.imwrite("face_detected001.png", face)
        return face

    def extract_features(self, face_image):
        """
        DeepFace yordamida feature extraction
        """
        embedding = DeepFace.represent(face_image, 
                                        model_name='VGG-Face',
                                        enforce_detection=False)
        return np.array(embedding[0]['embedding'])
    
    def load_dataset(self, dataset_path):
        """
        Datasetni yuklash va feature extract qilish
        """
        features = []
        labels = []
        print("Dataset yuklanmoqda...")
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            print(f"{person_name} uchun rasmlar ishlanmoqda...")
            
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                face = self.detect_and_extract_face(image_path)
                if face is not None:
                    feature = self.extract_features(face)
                    
                    if feature is not None:
                        features.append(feature)
                        labels.append(person_name)
                        
        print(f"Jami {len(features)} ta feature extract qilindi")
        
        return np.array(features), np.array(labels)
    
    def create_label_encoding(self, labels):
        """
        Label encoding yaratish
        """
        unique_labels = np.unique(labels)
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        encoded_labels = [self.label_encoder[label] for label in labels]
        return np.array(encoded_labels)
    
    def train_model(self, dataset_path, test_size=0.1):
        """
        Modelni o'qitish
        """
        features, labels = self.load_dataset(dataset_path)
        
        if len(features) == 0:
            raise ValueError("Dataset bo'sh yoki yuzlar aniqlanmadi")
        
        encoded_labels = self.create_label_encoding(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # SVM modelini o'qitish
        print("Model o'qitilmoqda...")
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=[self.reverse_label_encoder[i] for i in range(len(self.reverse_label_encoder))]))
        
        return accuracy
    
    def save_model(self, model_path):
        """
        Modelni saqlash
        """
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saqlandi: {model_path}")
    
    def load_model(self, model_path):
        """
        Modelni yuklash
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.reverse_label_encoder = model_data['reverse_label_encoder']
        
        print(f"Model yuklandi: {model_path}")
    
    def predict_image(self, image_path, threshold=0.5):
        """
        Bitta rasmni bashorat qilish
        """
        if self.model is None:
            raise ValueError("Model avval o'qitilishi yoki yuklanishi kerak")
        
        face = self.detect_and_extract_face(image_path)
        plt.savefig("face_detected.png")
        if face is None:
            return None, None, "Yuz aniqlanmadi"
        
        feature = self.extract_features(face)
        
        if feature is None:
            return None, None, "Feature extract qilib bo'lmadi"
        
        feature = feature.reshape(1, -1)
        prediction = self.model.predict(feature)[0]
        probability = self.model.predict_proba(feature)[0]
        
        max_prob = np.max(probability)
        
        if max_prob < threshold:
            return None, max_prob, "Noma'lum shaxs"
        
        predicted_person = self.reverse_label_encoder[prediction]
        
        return predicted_person, max_prob, "Muvaffaqiyatli"
    
    def predict_realtime(self, threshold=0.5):
        """
        Real-time prediction (webcam)
        """
        if self.model is None:
            raise ValueError("Model avval o'qitilishi yoki yuklanishi kerak")
        cap = cv2.VideoCapture(0)
        print("Real-time prediction boshlandi. 'q' tugmasi bilan chiqing.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            detections = self.detector.detect_faces(frame_rgb)
            
            for detection in detections:
                x, y, w, h = detection['box']
                
                face = frame_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                
                feature = self.extract_features(face_resized)
                if feature is not None:
                    feature = feature.reshape(1, -1)
                    prediction = self.model.predict(feature)[0]
                    probability = self.model.predict_proba(feature)[0]
                    max_prob = np.max(probability)
                    
                    if max_prob >= threshold:
                        predicted_person = self.reverse_label_encoder[prediction]
                        label = f"{predicted_person} ({max_prob:.2f})"
                        color = (0, 255, 0) 
                    else:
                        label = f"Unknown({max_prob:.2f})"
                        color = (0, 0, 255) 

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fr_system = FaceRecognitionSystem()
    # dataset_path = "facerec_datasets/Dataset/Faces"  
    # model_path = "models/deep_face_recognition_model.pkl"
    
    # print("Model o'qitilmoqda...")
    # accuracy = fr_system.train_model(dataset_path)
    
    # fr_system.save_model(model_path)
    
    # test_image = "tests/akshay-kumar-20191210145129-78.jpg" 
    # predicted_person, probability, status = fr_system.predict_image(test_image)
    # print(f"\nTest natijasi:")
    # print(f"Bashorat: {predicted_person}")
    # print(f"Ehtimollik: {probability:.4f}")
    # print(f"Status: {status}")

    # Real-time prediction
    # fr_system.predict_realtime()
    model_path = "models/deep_face_recognition_model.pkl"
    fr_system_loaded = FaceRecognitionSystem()
    fr_system_loaded.load_model(model_path)
    predicted_person, probability, status = fr_system_loaded.predict_image("tests/alexandra-daddario-2022-noms-450x600.jpg")
    print(f"Bashorat: {predicted_person}, Ehtimollik: {probability:.4f}")
    print(f"Status: {status}")

    # predicted_person, probability, status = fr_system_loaded.predict_image("tests/alexandra-daddario-2022-noms-450x600.jpg")
    # print(f"Bashorat: {predicted_person}, Ehtimollik: {probability:.4f}")
    # print(f"Status: {status}")
    """
    # Avval saqlangan modelni yuklash
    fr_system_loaded = FaceRecognitionSystem()
    fr_system_loaded.load_model(model_path)
    
    # Bashorat qilish
    predicted_person, probability, status = fr_system_loaded.predict_image("new_test_image.jpg")
    print(f"Bashorat: {predicted_person}, Ehtimollik: {probability:.4f}")
    """