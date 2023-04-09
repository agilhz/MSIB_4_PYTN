#!/usr/bin/env python
# coding: utf-8

# <h1>Project Overview</h1>

# Using what you've learned; download the NYC Property Sales Dataset from Kaggle. This dataset is a record of every building or building unit (apartment, etc.) sold in the New York City porperty market over a 12-month period.
# 
# This dataset contains the location, address, type, sale price, and sale date of building units sold. A reference on the trickier fields:
# 
# 1. BOROUGH : A digit code for the borough the property is located in; in order these are Manhattan (1), Bronx (2), Brooklyn (3), Queens (4), and Staten Island (5).
# 2. BLOCK; LOT :The combination of borough, block, and lot forms a unique key for property in New York City. Commonly called a BBL.
# 3. BUILDING CLASS AT PRESENT and BUILDING CLASS AT TIME OF SALE: : The type of building at various points in time.
# 
# Note that because this is a financial transaction dataset, there are some points that need to be kept in mind:
# 
# 1. Many sales occur with a nonsensically small dollar amount: $0 most commonly. These sales are actually transfers of deeds between parties: for example, parents transferring ownership to their home to a child after moving out for retirement.
# 2. This dataset uses the financial definition of a building/building unit, for tax purposes. In case a single entity owns the building in question, a sale covers the value of the entire building. In case a building is owned piecemeal by its residents (a condominium), a sale refers to a single apartment (or group of apartments) owned by some individual.

# <h1>Import Library</h1>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore")


# <h1>Dataset</h1>

# In[2]:


# read dataset
df = pd.read_csv('/Users/agilh/Downloads/nyc-rolling-sales.csv')


# menampilkan dataset
df


# In[3]:


# melihat informasi dari dataset
df.info()


# <h1>Preprocessing</h1>

# <h2>Mengecek Missing Value</h2>

# In[4]:


# mengantisipasi untuk unstandard missing value
missing_values = ['n/a', 'na', "--", "?", "NA", 'n-a', 'NaN',' ',' -  ']
df = pd.read_csv('/Users/agilh/Downloads/nyc-rolling-sales.csv', na_values=missing_values)


# In[5]:


# melihat jumlah missing value yang ada pada setiap kolom
df.isnull().sum()


# <h2>Drop Kolom</h2>

# In[6]:


# drop kolom yang tidak dibutuhkan untuk proses analisis
df.drop(labels=['Unnamed: 0','EASE-MENT','ADDRESS','APARTMENT NUMBER', 'ZIP CODE'], axis=1, inplace=True)


# <h2>Handling Missing Value</h2>

# In[7]:


# drop missing value
df.dropna(inplace=True)


# In[8]:


# melihat informasi dari dataset
df.info()


# <h2>Drop Duplikasi</h2>

# In[9]:


# menghitung jumlah data duplikat
sum(df.duplicated())


# In[10]:


# drop data duplikat
df = df.drop_duplicates()


# In[11]:


# menghitung jumlah data duplikat
sum(df.duplicated())


# In[12]:


# melihat informasi dari dataset
df.info()


# <h2>Perubahan Type Data</h2>

# In[13]:


# membuat variabel untuk pengkategorian
categoricals_columns = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 
                'YEAR BUILT', 'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE',
                'BUILDING CLASS AT PRESENT']
float_columns = ['SALE PRICE', 'LAND SQUARE FEET', 'GROSS SQUARE FEET']
integer_columns = ['BLOCK', 'LOT']

# ubah kolom yang seharusnya kategori menjadi tipe 'str'
for col in categoricals_columns:
    df[col] = df[col].astype('str')

# ubah kolom yang seharusnya float menjadi tipe 'float'
for col in float_columns:
    df[col] = df[col].astype('float64')

# ubah kolom yang seharusnya int menjadi tipe 'int'
for col in integer_columns:
    df[col] = df[col].astype('int64')


# In[14]:


# melihat informasi dari dataset
df.info()


# <h2>Menghapus Outlier</h2>

# In[15]:


# Hitung kuartil 1 dan kuartil 3 setiap kolom pada data
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

# Hitung IQR (interquartile range) setiap kolom pada data
IQR = Q3 - Q1

# Definisikan batas bawah dan batas atas setiap kolom pada data
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hapus data yang melebihi batas bawah atau batas atas pada setiap kolom
df_filtered = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

# Lakukan indexing ulang pada dataframe karena indexnya berantakan setelah dilakukan penghapusan data outliers
df_filtered = df_filtered.reset_index(drop=True)

# Tampilkan informasi tentang jumlah data sebelum dan sesudah penghapusan outliers
print(f"Jumlah data awal: {len(df)}")
print(f"Jumlah data setelah penghapusan outliers: {len(df_filtered)}")

df = df_filtered


# In[16]:


df


# <h1>Measure of Tendency Central and Measure of Spread</h1>

# In[17]:


# memunculkan semua hasil statistik describe
df.describe(include='all')


# <h2>Mean</h2>

# In[18]:


# mencari mean gross square feet
print("Mean GROSS SQUARE FEET = ", df['GROSS SQUARE FEET'].mean())


# <h2>Median</h2>

# In[19]:


# mencari median land square feet
print("Median LAND SQUARE FEET = ", df['LAND SQUARE FEET'].median())


# <h2>Modus</h2>

# In[20]:


# mencari modus building class category
cols = ['BUILDING CLASS CATEGORY']
for col in cols:
    print("Modus "+col+" = ",df[col].mode()[0])


# <h2>Range</h2>

# In[21]:


# mencari range sale price
print('Range SALE PRICE = ', (df['SALE PRICE'].max() - df['SALE PRICE'].min()))


# <h2>Variance</h2>

# In[22]:


# mencari variance block
print('Variance dari BLOCK = ', df['BLOCK'].var())


# <h2>Standard Deviation</h2>

# In[23]:


# mencari standard deviation dari LOT
print('Standard Deviation dari LOT = ', df['LOT'].std())


# <h1>Propability Distribution</h1>

# Implementasi Propablity Distribution digunakan untuk melihat distribusi dari kolom 'GROSS SQUARE FEET'

# <h2>Normal Distribution</h2>

# In[25]:


# menentukan style plot
sns.set_style('whitegrid')

# membuat histogram Gross Square Feet properti di New York City pada tahun 2016 dan 2017
sns.histplot(data=df, x='GROSS SQUARE FEET', bins=30, kde=True)
plt.title('Histogram Gross Square Feet properti di New York City pada tahun 2016 dan 2017', fontsize=14)
plt.xlabel('Gross Square Feet Properti', fontsize=14)
plt.ylabel('Frekuensi', fontsize=14)
plt.xticks(rotation=90)
plt.show()


# Dari visualisasi histogram diatas memiliki jenis skew ke kiri, untuk mengatasi hal tersebut perlu adanya transformasi data untuk memperbaikinya

# <h2>Handling Skew</h2>

# In[26]:


# Menghilangkan nilai 0 dari data
df = df[df['GROSS SQUARE FEET'] > 0]

# Melakukan sqrt-transform pada data
sqrt_data = np.sqrt(df['GROSS SQUARE FEET'])

# Membuat histogram dari sqrt-transformed data
sns.set_style('whitegrid')
sns.histplot(data=sqrt_data, bins=30, kde=True)
plt.title('Histogram sqrt-transformed Gross Square Feet properti di New York City pada tahun 2016 dan 2017', fontsize=14)
plt.xlabel('sqrt-transformed Gross Square Feet Properti', fontsize=14)
plt.ylabel('Frekuensi', fontsize=14)
plt.show()


# <h2>Mengecek Nilai Skew</h2>

# In[28]:


# Menghitung nilai skewed pada data sqrt-transformed
skew = sqrt_data.skew()

print("Nilai skewness pada data sqrt-transformed: {:.2f}".format(skew))


# <h1>Confidence Intervals</h1>

# Implementasi Confidence Intervals untuk melihat 'SALE PRICE' di New York City pada tahun 2016 dan 2017

# <h2>Mengganti Nilai 0 Dengan Nilai Median</h2>

# In[29]:


# Mencari nilai median dari kolom 'SALE PRICE'
median_price = df['SALE PRICE'].median()

# Mengganti nilai 0 dengan nilai median
df['SALE PRICE'] = df['SALE PRICE'].replace(0, median_price)


# <h2>Histogram Untuk Confidence Interval</h2>

# In[30]:


# ambil sample data dari dataset
sample_data = df.sample(n=1000, random_state=42)

# hitung mean dan standard deviation dari sample
mean = sample_data['SALE PRICE'].mean()
std_dev = sample_data['SALE PRICE'].std()

# hitung confidence interval
ci_low = mean - 1.96 * std_dev / np.sqrt(len(sample_data))
ci_high = mean + 1.96 * std_dev / np.sqrt(len(sample_data))

# buat histogram dengan garis vertikal menunjukkan confidence interval
sns.histplot(data=sample_data, x='SALE PRICE', kde=True)
plt.axvline(x=ci_low, color='black', linestyle='--')
plt.axvline(x=ci_high, color='black', linestyle='--')

# tambahkan label dan judul plot
plt.xlabel('Sale Price ($)')
plt.ylabel('Frequency')
plt.title('Histogram dengan Confidence Interval untuk Sale Price')

plt.show()


# <h1>Hypothesis Testing</h1>

# Apakah harga properti di Manhattan secara signifikan lebih tinggi daripada harga properti di borough lain di New York City?

# In[31]:


# filter data untuk Manhattan dan borough lainnya
manhattan = df[df['BOROUGH'] == 1]
other_boroughs = df[df['BOROUGH'].isin([2, 3, 4, 5])]

# melakukan two-sample t-test dengan mengasumsikan varian yang tidak sama
t, p = ttest_ind(manhattan['SALE PRICE'], other_boroughs['SALE PRICE'], equal_var=False)

# tentukan tingkat signifikansi
alpha = 0.05

# interpretasikan nilai p
if p < alpha:
    print('Tolak hipotesis: Harga properti di Manhattan secara signifikan lebih tinggi daripada harga properti di borough lainnya')
else:
    print('Terima hipotesis: Harga properti di Manhattan tidak secara signifikan lebih tinggi daripada harga properti di borough lainnya')


# # Overall Analysis

# 1. Kesimpulan dari Measure of Tendency Central and Measure of Spread yaitu :

# In[32]:


df.describe(include='all')


# 2. Propability Distribution digunakan untuk melihat distribusi dari kolom 'GROSS SQUARE FEET' berada di rentang 2400 ~ 2600
# 3. Confidence Intervals 'SALE PRICE' di New York City pada tahun 2016 dan 2017 berada di rentang sekitar 5.6 ~ 5.8 atau 560.000 ~ 580.000 US Dolar
# 4. Hipotesis testing yang muncul untuk pertanyaan *Apakah harga properti di Manhattan secara signifikan lebih tinggi daripada harga properti di borough lain di New York City?* menghasilkan hipotesis *Terima hipotesis: Harga properti di Manhattan tidak secara signifikan lebih tinggi daripada harga properti di borough lainnya*
