import cv2
import skfuzzy as fuzz
import numpy as np

# تشخیص چهره با استفاده از OpenCV
def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread("Image_Name.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)


    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# منطق فازی برای تصمیم‌گیری در مورد تشخیص چهره
def fuzzy_face_detection(faces_detected):
    # دسته‌بندی فازی ساده
    x = np.arange(0, 101, 1)
    face_detected = fuzz.trimf(x, [0, 50, 100])

    # مقداردهی داده‌های فازی
    face_detected_level = fuzz.interp_membership(x, face_detected, faces_detected)

    # تصمیم‌گیری بر اساس مقدار داده‌های فازی
    if face_detected_level >= 50:
        return "The face is detected."
    else:
        return "Face not detected."

# مثال تشخیص چهره و استفاده از منطق فازی
image_path = 'image.jpg'
detect_face(image_path)

# فرض می‌کنیم 5 چهره تشخیص داده شده است.
num_faces_detected = 5
result = fuzzy_face_detection(num_faces_detected)
print("The result of face recognition using fuzzy logic:", result)
