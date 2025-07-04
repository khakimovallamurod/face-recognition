import cv2
from mtcnn import MTCNN

def image_to_face(image_path, target_size=(224, 224)):
    """
    MTCNN yordamida rasmda yuzni aniqlash va ajratib olish
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = image.copy()
    # image rgb save 
    cv2.imwrite("image_rgb.png", image_rgb)
    print("Image RGB saved as image_rgb.png")
    detector = MTCNN()
    detections = detector.detect_faces(image_rgb)
    
    if len(detections) == 0:
        return None
    
    detection = max(detections, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = detection['box']
    
    face = image_rgb[y:y+h, x:x+w]
    
    face = cv2.resize(face, target_size)
    return face
def main():
    image_path = "tests/alia-bhatt-1.png"
    face = image_to_face(image_path)
    cv2.imwrite("face_detected.png", face)
main()