import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import csv
from collections import Counter
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# تنظيف جملة واحدة
def clean_caption(text):
    text = text.lower()  # حروف صغيرة
    text = text.translate(str.maketrans('', '', string.punctuation))  # إزالة punctuation
    tokens = word_tokenize(text)  # تقسيم الجملة لكلمات
    tokens = [t for t in tokens if t not in stopwords.words('english')]  # إزالة stopwords
    return tokens

# قراءة الملف csv وتنظيف جميع captions
def load_and_clean_captions(file_path):
    captions_dict = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # تجاهل العنوان
        for img_name, caption in reader:
            tokens = clean_caption(caption)
            captions_dict.setdefault(img_name, []).append(tokens)
    return captions_dict

# بناء قائمة الكلمات المفلترة (vocabulary)
def build_vocabulary(captions_dict, min_freq=2):
    all_tokens = []
    for caps in captions_dict.values():
        for tokens in caps:
            all_tokens.extend(tokens)
    counter = Counter(all_tokens)
    vocab = [word for word, freq in counter.items() if freq >= min_freq]
    return vocab

# تحويل captions إلى binary vectors
def captions_to_binary_vectors(captions_dict, vocab):
    word2idx = {word: i for i, word in enumerate(vocab)}
    labels = {}
    for img_name, caps_list in captions_dict.items():
        img_labels = np.zeros(len(vocab), dtype=int)
        for tokens in caps_list:
            for t in tokens:
                if t in word2idx:
                    img_labels[word2idx[t]] = 1
        labels[img_name] = img_labels
    return labels
