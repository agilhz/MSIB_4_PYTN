#!/usr/bin/env python
# coding: utf-8

# <h1>Project overview</h1>
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. There are four datasets:
# 
# 1. bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
# 
# 2. bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
# 
# 3. bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
# 
# 4. bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
# 
# The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM). The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

# <h1>Import Libraries</h1>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


# <h1>Dataset</h1>

# In[2]:


# read dataset
df = pd.read_csv('/Users/agilh/Downloads/bank-additional-full.csv', sep=';')


# menampilkan dataset
df


# In[3]:


# melihat informasi dari dataset
df.info()


# <h1>Preprocessing</h1>

# <h2>Mengecek Missing Value</h2>

# In[4]:


# melihat jumlah missing value yang ada pada setiap kolom
df.isnull().sum()


# In[5]:


# menampilkan 50 data teratas untuk melihat unstandard missing value
df.head(50)


# In[6]:


# manampilkan jumlah unstandard missing value 'unknown'
missing_values = ['unknown']
df = pd.read_csv('/Users/agilh/Downloads/bank-additional-full.csv', sep=';', na_values=missing_values)


# In[7]:


df.isnull().sum()


# In[8]:


# menampilkan 50 data teratas
df.head(50)


# <h2>Handling Missing Value</h2>

# In[9]:


# mengganti value unstandard missing value 'unknown' manjadi NaN
df.replace(['unknown'], np.nan, inplace=True)


# In[10]:


# melihat kembali jumlah unstandard missing value 'unknown'
df[
    df=='unknown'
].count().sort_values(ascending=False)


# In[11]:


# menampilkan 50 data teratas
df.head(50)


# In[12]:


# Mengganti missing value di kolom numerikal dengan mean
numerical_columns = list(df.select_dtypes(include=['int64', 'float64']).columns.values)
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Mengganti missing value di kolom kategorikal dengan median
categorical_columns = list(df.select_dtypes(include=['object']).columns.values)
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])


# In[13]:


# menampilkan 50 data teratas untuk memastikan tidak ada lagi missing value
df.head(50)


# In[14]:


# manampilkan jumlah missing value
df.isnull().sum()


# <h1>Analisis Data Dengan Visualisasi</h1>

# 1. Berapa total hasil seseorang tertarik untuk Subscribe Deposito?

# In[15]:


# Plot histogram of y (subscribe deposito)
plt.hist(df["y"])
plt.xlabel("y")
plt.ylabel("Frekuensi")
plt.title("Frekuensi Subscribe Deposito")
plt.show()


# Dilihat dari visualisasi diatas menunjukkan bahwa kebanyakan orang tidak tertarik untuk melakukan subscribe deposito

# 2. Bagaimana persentase pelanggan yang berlangganan deposito berdasarkan kategori usia?

# In[16]:


# Membuat bins untuk kategori usia
bins = [0, 19, 29, 39, 49, 59, 69, 120]
labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Menghitung jumlah pelanggan yang berlangganan deposito berdasarkan kategori usia
grouped = df[df["y"] == "yes"].groupby(pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True))["y"].count()

# Membuat pie chart
plt.pie(grouped, labels=grouped.index, autopct="%1.1f%%")
plt.title("Persentase Pelanggan yang Berlangganan Deposito Berdasarkan Kategori Usia")
plt.show()


# Hasil yang didapatkan dari visualisasi diatas adalah persentase pelanggan yang melakukan langganan deposito paling banyak di terjadi di kategori usia 30-39 dengan persentase 37% disusul oleh kategori usia 40-49 dengan persentase 18%

# <h1>Preprocessing Dataset sebelum Modelling</h1>

# <h2>Encoding Variabel Kategorikal</h2>

# In[17]:


# List kolom yang di encoder
columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Buat objek encoder
encoder = OneHotEncoder()

# Fit dan transform data
encoded_data = encoder.fit_transform(df[columns_to_encode]).toarray()

# Buat dataframe baru dengan data yang sudah di-encode
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns_to_encode))

# Gabungkan data set awal dengan data yang sudah di-encode
df_encoded = pd.concat([df, encoded_df], axis=1)

# Hapus kolom-kolom yang sudah di-encode
df_encoded = df_encoded.drop(columns_to_encode, axis=1)


# <h2>Scalling Variabel Numerikal</h2>

# In[18]:


# list kolom yang akan di scaling
columns_to_scale = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# inisiasi MinMaxScaler
scaler = MinMaxScaler()

# lakukan fit transform pada data numerik yang telah di-encode
df[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])


# <h1>Modeling</h1>

# In[19]:


# definisikan fitur X dan target y
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# bagi data menjadi training set dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# buat objek untuk masing-masing algoritma
logreg = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
nb = GaussianNB()

# latih model pada training set
logreg.fit(X_train, y_train)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
nb.fit(X_train, y_train)

# prediksi target pada testing set
y_pred_logreg = logreg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_nb = nb.predict(X_test)

# hitung akurasi masing-masing model
acc_logreg = accuracy_score(y_test, y_pred_logreg)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_svm = accuracy_score(y_test, y_pred_svm)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_nb = accuracy_score(y_test, y_pred_nb)

# tampilkan akurasi masing-masing model
print('Akurasi Logistic Regression:', acc_logreg)
print('Akurasi KNN:', acc_knn)
print('Akurasi SVM:', acc_svm)
print('Akurasi Decision Tree:', acc_dt)
print('Akurasi Random Forest:', acc_rf)
print('Akurasi Naive Bayes:', acc_nb)


# Analisis :
# Model Random Forest tampil memberikan rata-rata akurasi tinggi

# In[20]:


# hitung confusion matrix dari masing-masing model
cm_rf = confusion_matrix(y_test, y_pred_rf)

# tampilkan akurasi model
print('Akurasi Random Forest:', acc_rf)

# tampilkan confusion matrix
print('Confusion Matrix Random Forest:')
print(cm_rf)

# hasil klasifikasi
print(metrics.classification_report(y_test,y_pred_rf))


# Untuk proses klasifikasi saya memilih menggunakan algoritma Random Forest karena performa yang dihasilkan lebih baik dari yang algoritma yang lain seperti Logistic Regression dan KNN. Hasil classification diatas dapat dilihat menghasilkan accuracy sebesar 0.91.
