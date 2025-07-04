
import os
import cv2
import numpy as np
from mtcnn import MTCNN
import shutil
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt

class DatasetPreparator:
    def __init__(self):
        self.detector = MTCNN()
        
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                              rotate_limit=10, p=0.3),
        ])
    
    def extract_faces_from_video(self, video_path, output_dir, person_name):
        """Videodan yuzlarni ajratib olish"""
        os.makedirs(os.path.join(output_dir, person_name), exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        face_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 5 != 0:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(rgb_frame)
            
            for i, result in enumerate(results):
                if result['confidence'] > 0.95: 
                    x, y, w, h = result['box']
                    
                    margin_x = int(w * 0.6)  
                    margin_y_top = int(h * 0.4)  
                    margin_y_bottom = int(h * 0.8)  

                    crop_x = max(0, x - margin_x)
                    crop_y = max(0, y - margin_y_top)
                    crop_w = min(frame.shape[1] - crop_x, w + 2*margin_x)
                    crop_h = min(frame.shape[0] - crop_y, h + margin_y_top + margin_y_bottom)
                    face = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    if face.shape[0] > 50 and face.shape[1] > 50:  # Minimal o'lcham
                        face_path = os.path.join(output_dir, person_name, 
                                               f"{person_name}_{face_count:04d}.jpg")
                        cv2.imwrite(face_path, face)
                        face_count += 1
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
           

            print(f"Frame: {frame_count}, Faces: {face_count}", end='\r')

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{person_name}: {face_count} ta yuz ajratildi")
        return face_count
    
    def extract_faces_from_images(self, images_dir, output_dir, person_name):
        """Rasmlardan yuzlarni ajratib olish"""
        os.makedirs(os.path.join(output_dir, person_name), exist_ok=True)
        
        face_count = 0
        
        for img_name in os.listdir(images_dir):
            img_path = os.path.join(images_dir, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.detector.detect_faces(rgb_img)
                
                for i, result in enumerate(results):
                    if result['confidence'] > 0.9:
                        x, y, w, h = result['box']
                        margin = 20
                        x = max(0, x - margin)
                        y = max(0, y - margin)
                        w = min(img.shape[1] - x, w + 2*margin)
                        h = min(img.shape[0] - y, h + 2*margin)
                        
                        face = img[y:y+h, x:x+w]
                        
                        if face.shape[0] > 80 and face.shape[1] > 80:
                            face_path = os.path.join(output_dir, person_name,
                                                   f"{person_name}_{face_count:04d}.jpg")
                            cv2.imwrite(face_path, face)
                            face_count += 1
                            
            except Exception as e:
                print(f"Xatolik {img_path}: {e}")
                continue
        
        print(f"{person_name}: {face_count} ta yuz ajratildi")
        return face_count
    
    def augment_dataset(self, input_dir, output_dir, target_count=100):
        """Dataset augmentation qilish"""
        os.makedirs(output_dir, exist_ok=True)
        
        for person_name in os.listdir(input_dir):
            person_input = os.path.join(input_dir, person_name)
            person_output = os.path.join(output_dir, person_name)
            
            if not os.path.isdir(person_input):
                continue
            
            os.makedirs(person_output, exist_ok=True)
            
            images = os.listdir(person_input)
            for img_name in images:
                src = os.path.join(person_input, img_name)
                dst = os.path.join(person_output, img_name)
                shutil.copy2(src, dst)
            
            current_count = len(images)
            
            if current_count < target_count:
                need_count = target_count - current_count
                
                print(f"{person_name}: {current_count} -> {target_count}")
                
                for i in range(need_count):
                    random_img = np.random.choice(images)
                    img_path = os.path.join(person_input, random_img)
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        augmented = self.augmentation(image=rgb_img)['image']
                        bgr_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                        
                        aug_path = os.path.join(person_output, 
                                              f"{person_name}_aug_{i:04d}.jpg")
                        cv2.imwrite(aug_path, bgr_img)
                        
                    except Exception as e:
                        continue
    
    def split_dataset(self, input_dir, output_dir, test_size=0.1, val_size=0.2):
        """Datasetni train/test/validation ga bo'lish"""
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'validation')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for person_name in os.listdir(input_dir):
            person_path = os.path.join(input_dir, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            images = os.listdir(person_path)
            
            if len(images) < 3:
                print(f"Warning: {person_name} da kam rasm ({len(images)})")
                continue

            # Train/validation split
            train_imgs, val_imgs = train_test_split(
                images, test_size=val_size, random_state=42
            )
            
           
            os.makedirs(os.path.join(train_dir, person_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, person_name), exist_ok=True)

            # Fayllarni ko'chirish
            for img in train_imgs:
                src = os.path.join(person_path, img)
                dst = os.path.join(train_dir, person_name, img)
                shutil.copy2(src, dst)

            for img in val_imgs:
                src = os.path.join(person_path, img)
                dst = os.path.join(val_dir, person_name, img)
                shutil.copy2(src, dst)

            print(f"{person_name}: Train={len(train_imgs)}, "
                  f"Val={len(val_imgs)}")

    def validate_dataset(self, dataset_dir):
        """Dataset validatsiyasi"""
        print("Dataset validatsiyasi...")
        
        total_images = 0
        person_count = 0
        
        for person_name in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            images = os.listdir(person_path)
            image_count = len([img for img in images 
                             if img.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            print(f"{person_name}: {image_count} rasm")
            
            total_images += image_count
            person_count += 1
            
            # Kamida 5 ta rasm bo'lishi kerak
            if image_count < 5:
                print(f"  Warning: Kam rasm!")
        
        print(f"\nJami: {person_count} kishi, {total_images} rasm")
        print(f"O'rtacha: {total_images/person_count:.1f} rasm/kishi")

def main():
    prep = DatasetPreparator()
    
    # 1. Videodan yuzlarni ajratish
    # prep.extract_faces_from_video(2, "raw_faces", "muzaffar")
    
    # 2. Rasmlardan yuzlarni ajratish
    # prep.extract_faces_from_images("person_photos", "raw_faces", "person1")
    
    # 3. Dataset augmentation
    # prep.augment_dataset("raw_faces", "augmented_faces", target_count=50)
    
    # 4. Dataset bo'lish
    # prep.split_dataset("peopledatasets", "person_dataset",
    #                    test_size=0.0, val_size=0.1)
    
    # 5. Validatsiya
    # prep.validate_dataset("face_dataset/train")
    

if __name__ == "__main__":
    main()