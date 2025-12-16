from src.preprocess_images import load_and_preprocess_images
from src.preprocess_captions import load_and_clean_captions, build_vocabulary, captions_to_binary_vectors

# 🔹 معالجة الصور
images = load_and_preprocess_images("data/Images", size=(224,224), limit=500)
print("Number of images loaded:", len(images))

# 🔹 معالجة captions
captions = load_and_clean_captions("data/captions.txt")
print("Number of captions loaded:", len(captions))

# 🔹 بناء vocabulary
vocab = build_vocabulary(captions, min_freq=2)
print("Vocabulary size:", len(vocab))
print("Sample words:", vocab[:20])

# 🔹 تحويل captions إلى binary vectors
labels = captions_to_binary_vectors(captions, vocab)
first_img = list(captions.keys())[0]
print("First image:", first_img)
print("Captions tokens:", captions[first_img])
print("Binary vector example:", labels[first_img])
