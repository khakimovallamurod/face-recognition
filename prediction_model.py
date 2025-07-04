
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras_facenet import FaceNet
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class OptimizedFaceRecognitionSystem:
    def __init__(self):
        self.detector = MTCNN()
        
        self.facenet_model = None
        self.load_facenet_model()
        
        self.svm_model = None
        self.rf_model = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.min_confidence = 0.95  # MTCNN uchun
        self.min_samples_per_class = 3
        
    def load_facenet_model(self):
        """FaceNet modelini yuklash"""
        try:
            self.facenet_model = FaceNet()
            print("✓ FaceNet model muvaffaqiyatli yuklandi")
        except Exception as e:
            print(f"✗ FaceNet model yuklanmadi: {e}")
    
    def preprocess_face(self, img, box):
        """Yuzni preprocessing qilish - yaxshilangan versiya"""
        x, y, w, h = box
        
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        face = img[y:y+h, x:x+w]
        
        # Face alignment (asosiy)
        face = self.align_face(face)
        
        # 160x160 ga o'zgartirish (FaceNet uchun)
        face = cv2.resize(face, (160, 160))
        
        # Histogram equalization
        if len(face.shape) == 3:
            face_yuv = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
            face_yuv[:,:,0] = cv2.equalizeHist(face_yuv[:,:,0])
            face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
        
        face = face.astype('float32')
        
        # Normalizatsiya (FaceNet uchun standart)
        face = (face - 127.5) / 128.0
        
        return face
    
    def align_face(self, face):
        """Yuzni to'g'rilash (asosiy versiya)"""
        # Bu yerda keypoint-lar orqali alignment qilish mumkin
        # Hozircha asosiy blur reduction va sharpening
        face = cv2.GaussianBlur(face, (3, 3), 0)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        face = cv2.filter2D(face, -1, kernel)
        return face
    
    def detect_faces(self, img):
        """Rasmdagi yuzlarni aniqlash - yaxshilangan"""
        if img is None:
            return [], []
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Multi-scale detection
        results = self.detector.detect_faces(rgb_img)
        
        faces = []
        boxes = []
        
        for result in results:
            confidence = result['confidence']
            if confidence > self.min_confidence:
                box = result['box']
                
                # Box validatsiyasi
                x, y, w, h = box
                if w > 30 and h > 30:  # Minimum yuz o'lchami
                    try:
                        face = self.preprocess_face(rgb_img, box)
                        if face is not None and face.size > 0:
                            faces.append(face)
                            boxes.append(box)
                    except Exception as e:
                        continue
        
        return faces, boxes
    
    def get_face_embedding(self, face):
        """Yuz uchun embedding olish"""
        if self.facenet_model is None:
            return None
            
        try:
            face = np.expand_dims(face, axis=0)
            embedding = self.facenet_model.embeddings(face)
            return embedding[0]
        except Exception as e:
            print(f"Embedding olishda xatolik: {e}")
            return None
    
    def augment_image(self, img):
        """Data augmentation"""
        augmented = []
        
        # Original
        augmented.append(img)
        
        # Horizontal flip
        augmented.append(cv2.flip(img, 1))
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        augmented.append(bright)
        
        dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
        augmented.append(dark)
        
        # Rotation (kichik burchaklar)
        rows, cols = img.shape[:2]
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            augmented.append(rotated)
        
        return augmented
    
    def load_dataset(self, dataset_path, use_augmentation=True):
        """Dataset yuklash va embeddings olish - yaxshilangan"""
        embeddings = []
        labels = []
        
        print("Dataset yuklanmoqda...")
        
        person_count = {}
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            print(f"Processing: {person_name}")
            person_embeddings = []
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Augmentation agar kerak bo'lsa
                    images_to_process = [img]
                    if use_augmentation:
                        images_to_process = self.augment_image(img)
                    
                    for processed_img in images_to_process:
                        faces, _ = self.detect_faces(processed_img)
                        
                        for face in faces:
                            embedding = self.get_face_embedding(face)
                            if embedding is not None:
                                person_embeddings.append(embedding)
                                
                except Exception as e:
                    print(f"Xatolik: {img_path} - {e}")
            
            # Minimum sample check
            if len(person_embeddings) >= self.min_samples_per_class:
                embeddings.extend(person_embeddings)
                labels.extend([person_name] * len(person_embeddings))
                person_count[person_name] = len(person_embeddings)
            else:
                print(f"⚠️ {person_name} uchun kam sample ({len(person_embeddings)}), o'tkazib yuborildi")
        
        print(f"\nDataset statistikasi:")
        for person, count in person_count.items():
            print(f"  {person}: {count} samples")
        
        return np.array(embeddings), np.array(labels)
    
    def train_classifier(self, dataset_path, test_size=0.2):
        """Klassifikatorni o'rgatish - yaxshilangan"""
        print("Klassifikator o'rgatilmoqda...")
        
        # Dataset yuklash
        embeddings, labels = self.load_dataset(dataset_path, use_augmentation=True)
        
        if len(embeddings) == 0:
            print("Dataset bo'sh!")
            return False
        
        # Label encoding
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Feature scaling
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings_scaled, encoded_labels, 
            test_size=test_size, 
            stratify=encoded_labels,
            random_state=42
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Multiple classifiers bilan experiment
        models = self.train_multiple_models(X_train, y_train)
        
        # Best model tanlash
        best_score = 0
        best_model_name = ""
        
        for name, model in models.items():
            score = model.score(X_val, y_val)
            print(f"{name} validation accuracy: {score:.4f}")
            
            if score > best_score:
                best_score = score
                self.best_model = model
                best_model_name = name
        
        print(f"\n✓ Eng yaxshi model: {best_model_name} (accuracy: {best_score:.4f})")
        
        # Detailed evaluation
        self.evaluate_model(X_val, y_val)
        
        # Model saqlash
        self.save_model()
        
        print(f"O'rgatish tugadi. {len(set(labels))} kishi, {len(embeddings)} rasm")
        return True
    
    def train_multiple_models(self, X_train, y_train):
        """Turli xil modellarni o'rgatish"""
        models = {}
        
        print("\nTurli modellar o'rgatilmoqda...")
        
        # SVM with hyperparameter tuning
        print("1. SVM tuning...")
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        svm_grid = GridSearchCV(SVC(probability=True), svm_params, cv=3, scoring='accuracy')
        svm_grid.fit(X_train, y_train)
        models['SVM'] = svm_grid.best_estimator_
        self.svm_model = svm_grid.best_estimator_
        
        # Random Forest with hyperparameter tuning
        print("2. Random Forest tuning...")
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy')
        rf_grid.fit(X_train, y_train)
        models['Random Forest'] = rf_grid.best_estimator_
        self.rf_model = rf_grid.best_estimator_
        
        return models
    
    def evaluate_model(self, X_val, y_val):
        """Modelni batafsil baholash"""
        y_pred = self.best_model.predict(X_val)
        
        # Classification report
        print("\n=== Classification Report ===")
        class_names = self.label_encoder.classes_
        print(classification_report(y_val, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self):
        """Modelni saqlash"""
        model_data = {
            'best_model': self.best_model,
            'svm_model': self.svm_model,
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'min_confidence': self.min_confidence
        }
        
        with open('optimized_face_recognition_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✓ Model saqlandi: optimized_face_recognition_model.pkl")
    
    def load_model(self):
        """Modelni yuklash"""
        try:
            with open('optimized_face_recognition_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_model = model_data['best_model']
            self.svm_model = model_data.get('svm_model')
            self.rf_model = model_data.get('rf_model')
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.min_confidence = model_data.get('min_confidence', 0.95)
            
            print("✓ Model muvaffaqiyatli yuklandi")
            return True
        except Exception as e:
            print(f"✗ Model yuklanmadi: {e}")
            return False
    
    def predict_face(self, img, confidence_threshold=0.7):
        """Yuzni bashorat qilish - yaxshilangan"""
        if self.best_model is None:
            print("⚠️ Model yuklanmagan! Avval modelni o'rgating yoki yuklang.")
            return []
            
        faces, boxes = self.detect_faces(img)
        
        results = []
        
        for i, face in enumerate(faces):
            embedding = self.get_face_embedding(face)
            
            if embedding is not None:
                try:
                    # Feature scaling (agar scaler fit qilingan bo'lsa)
                    if hasattr(self.scaler, 'mean_'):
                        embedding_scaled = self.scaler.transform([embedding])
                    else:
                        # Agar scaler fit qilinmagan bo'lsa, original embedding ishlatamiz
                        embedding_scaled = [embedding]
                    
                    # Bashorat
                    prediction = self.best_model.predict(embedding_scaled)[0]
                    probabilities = self.best_model.predict_proba(embedding_scaled)[0]
                    confidence = probabilities.max()
                    
                    # Threshold check
                    if confidence >= confidence_threshold:
                        # Label decode
                        person_name = self.label_encoder.inverse_transform([prediction])[0]
                        
                        results.append({
                            'box': boxes[i],
                            'name': person_name,
                            'confidence': confidence,
                            'all_probabilities': dict(zip(self.label_encoder.classes_, probabilities))
                        })
                    else:
                        results.append({
                            'box': boxes[i],
                            'name': 'Unknown',
                            'confidence': confidence,
                            'all_probabilities': {}
                        })
                        
                except Exception as e:
                    print(f"Bashorat qilishda xatolik: {e}")
                    continue
        
        return results
    
    def test_model(self, test_dataset_path, detailed=True):
        """Modelni test qilish - yaxshilangan"""
        print("Model test qilinmoqda...")
        
        true_labels = []
        predicted_labels = []
        confidences = []
        person_results = {}
        
        for person_name in os.listdir(test_dataset_path):
            person_path = os.path.join(test_dataset_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            person_correct = 0
            person_total = 0
            
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    results = self.predict_face(img)
                    
                    if results:
                        predicted_name = results[0]['name']
                        confidence = results[0]['confidence']
                        
                        true_labels.append(person_name)
                        predicted_labels.append(predicted_name)
                        confidences.append(confidence)
                        
                        person_total += 1
                        if predicted_name == person_name:
                            person_correct += 1
                        elif detailed:
                            print(f"✗ Xato: {person_name} -> {predicted_name} (conf: {confidence:.3f})")
                    
                except Exception as e:
                    continue
            
            if person_total > 0:
                person_accuracy = person_correct / person_total
                person_results[person_name] = {
                    'accuracy': person_accuracy,
                    'correct': person_correct,
                    'total': person_total
                }
        
        if len(true_labels) > 0:
            overall_accuracy = accuracy_score(true_labels, predicted_labels)
            avg_confidence = np.mean(confidences)
            
            print(f"\n=== Test Natijalari ===")
            print(f"Umumiy aniqlik: {overall_accuracy:.4f}")
            print(f"O'rtacha ishonch: {avg_confidence:.4f}")
            print(f"Test qilingan rasmlar: {len(true_labels)}")
            
            # Per-person results
            print(f"\n=== Shaxsiy natijalar ===")
            for person, result in person_results.items():
                print(f"{person}: {result['correct']}/{result['total']} ({result['accuracy']:.3f})")
        
        return person_results
    
    def visualize_embeddings(self, dataset_path):
        """Embeddings ni visualizatsiya qilish (t-SNE)"""
        from sklearn.manifold import TSNE
        
        print("Embeddings visualizatsiya qilinmoqda...")
        
        embeddings, labels = self.load_dataset(dataset_path, use_augmentation=False)
        
        if len(embeddings) == 0:
            print("Dataset bo'sh!")
            return
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[color], label=label, alpha=0.7)
        
        plt.legend()
        plt.title('Face Embeddings Visualization (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("=== Optimizatsiya qilingan Face Recognition tizimi ===\n")
    
    fr_system = OptimizedFaceRecognitionSystem()
    
    # Modelni yuklashga harakat qilish
    model_loaded = fr_system.load_model()
    
    # 1. Modelni o'rgatish
    # dataset_path = "face_dataset/train"
    # if os.path.exists(dataset_path):
    #     print("Model o'rgatilmoqda...")
    #     success = fr_system.train_classifier(dataset_path)
    #     if not success:
    #         print("Model o'rgatilmadi!")
    #         return
            
    #     # Embeddings visualizatsiya
    #     try:
    #         fr_system.visualize_embeddings(dataset_path)
    #     except Exception as e:
    #         print(f"Visualizatsiya xatoligi: {e}")
    # else:
    #     print(f"❌ Training dataset topilmadi: {dataset_path}")
    #     print("Model ham mavjud emas!")
    #     return
    
    # 2. Test qilish
    test_path = "face_dataset/validation"
    if os.path.exists(test_path):
        print("\n" + "="*50)
        try:
            person_results = fr_system.test_model(test_path, detailed=True)
        except Exception as e:
            print(f"Test qilishda xatolik: {e}")
    else:
        print(f"⚠️ Test dataset topilmadi: {test_path}")
    
    if fr_system.best_model is not None:
        run_realtime = input("\nReal-time test ishga tushirishni xohlaysizmi? (y/n): ")
        if run_realtime.lower() == 'y':
            # Kamera raqamini tekshirish
            camera_id = 2
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print("Kamera ochilmadi, boshqa ID sinab ko'rilmoqda...")
                for i in range(1, 4):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        camera_id = i
                        print(f"Kamera {i} ishlatilmoqda")
                        break
                else:
                    print("Hech qanday kamera topilmadi!")
                    return
            
            print(f"Real-time test (Kamera {camera_id}). 'q' tugmasi bilan chiqish.")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame o'qilmadi!")
                    break
                
                frame_count += 1
                
                if frame_count % 3 == 0:
                    try:
                        results = fr_system.predict_face(frame, confidence_threshold=0.6)
                        
                        # Natijalarni chizish
                        for result in results:
                            box = result['box']
                            name = result['name']
                            confidence = result['confidence']
                            
                            x, y, w, h = box
                            
                            color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                            
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            
                            text = f"{name} ({confidence:.2f})"
                            cv2.putText(frame, text, (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    except Exception as e:
                        print(f"Frame processing xatoligi: {e}")
                        continue
                
                cv2.imshow('Optimized Face Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("❌ Model mavjud emas, real-time test ishlatilmaydi!")

if __name__ == "__main__":
    main()