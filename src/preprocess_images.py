import os
import cv2
import numpy as np

def load_and_preprocess_images(folder_path, size=(224, 224), limit=None):
    images = {}
    files = os.listdir(folder_path)
    if limit:
        files = files[:limit]
    
    for img_name in files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, size)      # تغيير الحجم
        img = img / 255.0                # normalization (0-1)
        images[img_name] = img
    return images
