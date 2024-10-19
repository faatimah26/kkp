import re
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string

# Membaca dataset
file_path = 'DATASET CYBERBULLYING INSTAGRAM - FINAL.xlsx'  # ganti dengan nama file yang sesuai
if os.path.exists(file_path):
    data_sentimen = pd.read_excel(file_path)
    print("Data berhasil dimuat!")
else:
    print("File tidak ditemukan, silakan periksa jalurnya.")

def remove_punctuation_and_number(text):
    for sp in string.punctuation:
        text = text.replace(sp, " ")
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'\s+', ' ', text)  # Menghapus spasi berlebih
    return text.strip()  # Menghapus spasi di awal dan akhir

# Menghapus tanda baca dan angka dari kolom 'Komentar'
data_sentimen['Komentar'] = data_sentimen['Komentar'].apply(remove_punctuation_and_number)

# Daftar stopwords
sw = ["bacod","cokk","anjirrr","anjeng","anjir","anying","anjay","kolot","tolol"]

# Fungsi untuk menghapus stopwords
def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    text = " ".join(text)
    text = re.sub(r'(.+?)\1+', r'\1', text)  # Menghapus pengulangan
    return text

# Menghapus stopwords dari kolom 'Komentar'
data_sentimen['Komentar'] = data_sentimen['Komentar'].apply(stopwords)

# Inisialisasi stopword remover
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

# Menghapus stopwords dengan Sastrawi (jika ingin)
data_sentimen['Komentar'] = data_sentimen['Komentar'].apply(stopword.remove)

# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk stemming
def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

# Melakukan stemming pada kolom 'Komentar'
data_sentimen['Komentar'] = data_sentimen['Komentar'].apply(stemming)

x = data_sentimen['Komentar']
y = data_sentimen['Kategori']
from collections import Counter

print(Counter(y))
sns.countplot(x=y)

from wordcloud import WordCloud, STOPWORDS
Bullying= " ".join(review for review in data_sentimen[data_sentimen['Kategori'] == 'Bullying'].Komentar)
Non_Bullying= " ".join(review for review in data_sentimen[data_sentimen['Kategori'] == 'Non-bullying'].Komentar)
stopwords = set(STOPWORDS)
def plot_cloud(wordcloud):
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud)
    plt.axis("off");

wordcloud1 = WordCloud(width = 3000, height = 2000, random_state=3, background_color='pink', colormap='Set2', collocations=False, stopwords = STOPWORDS).generate(Bullying)
plot_cloud(wordcloud1)

# replace label pada dataset
data_sentimen.Kategori.replace("Bullying", 0 , inplace = True)
data_sentimen.Kategori.replace("Non-bullying", 1 , inplace = True)
data_sentimen.Kategori = data_sentimen.Kategori.astype(int)
data_sentimen.head(10)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(data_sentimen['Komentar'],data_sentimen['Kategori'],test_size=0.3)

Count_vect = CountVectorizer(max_features=5000)
Count_vect.fit(data_sentimen['Komentar'])
Train_X_Count = Count_vect.transform(Train_X)
Test_X_Count = Count_vect.transform(Test_X)

"""LOGISTIC REGGRESSION"""

# Logistic Regression
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Setelah bagian ini, di mana Anda telah melatih model
# Logistic Regression
LR = LogisticRegression()
LR.fit(Train_X_Count, Train_Y)
score = LR.score(Test_X_Count, Test_Y)
score_t = LR.score(Train_X_Count, Train_Y)

print("Logistic Regression Accuracy Score Training-> {:.2f}%".format(score_t * 100))
print("Logistic Regression Accuracy Score Testing-> {:.2f}%".format(score * 100))


# Confusion matrix Logistic Regression pada data train
conf_matLR_t = confusion_matrix(Train_Y, LR.predict(Train_X_Count))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matLR_t, annot=True, fmt='d', 
            xticklabels=["Bullying", "Non-Bullying"], yticklabels=["Bullying", "Non-Bullying"])
plt.title('Confusion Matrix Logistic Regression pada Data Training')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')  # Menyimpan sebagai file
plt.close() 
# Klasifikasi report Logistic Regression pada data training
print("Klasifikasi Report Logistic Regression pada Data Training")
print(classification_report(Train_Y, LR.predict(Train_X_Count), target_names=["Bullying", "Non-Bullying"]))
print("\n")

# Klasifikasi report Logistic Regression pada data testing
print("Klasifikasi Report Logistic Regression pada Data Testing")
print(classification_report(Test_Y, LR.predict(Test_X_Count), target_names=["Bullying", "Non-Bullying"]))

# Simpan model dan vectorizer setelah pelatihan
with open('model.pkl', 'wb') as model_file:
    pickle.dump(LR, model_file)

with open('vectorizer.pkl', 'wb') as vect_file:
    pickle.dump(Count_vect, vect_file)

print("Model dan vectorizer berhasil diinisialisasi dan dilatih.")
