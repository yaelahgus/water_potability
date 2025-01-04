import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.write("AGUS ARIFUDIN")
st.write("\nA11.2021.13736")

# 1. Pengumpulan Data
water_data = pd.read_csv('data/water_potability.csv')

# 2. Menelaah Data
st.write("Informasi Dataset:")
st.write(water_data.describe())
st.write("\nJumlah Baris dan Kolom:", water_data.shape)
st.write("\nJumlah Nilai Unik Tiap Kolom:")
st.write(water_data.nunique())
st.write("\nCek Missing Values:")
st.write(water_data.isnull().sum())

# 3. Imputasi Missing Values
water_data.fillna(water_data.mean(), inplace=True)

# 4. Visualisasi Data
st.subheader('Distribusi Kelayakan Minum (Potability)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='Potability', data=water_data, ax=ax)
st.pyplot(fig)

# Korelasi antar fitur
st.subheader('Korelasi Antar Fitur')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(water_data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Histogram Plot
st.subheader('Distribusi Atribut')
fig, ax = plt.subplots(figsize=(12, 10))
water_data.hist(figsize=(12, 10), bins=20, ax=ax)
plt.suptitle('Distribusi Atribut')
st.pyplot(fig)

# 5. Pemodelan
X = water_data.drop('Potability', axis=1)
y = water_data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

### 6. Model Sebelum Normalisasi
st.subheader("Evaluasi Sebelum Normalisasi")

# Logistic Regression
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)
y_pred_log_before = log_model.predict(X_test)
accuracy_log_before = accuracy_score(y_test, y_pred_log_before)
f1_log_before = f1_score(y_test, y_pred_log_before)
precision_log_before = precision_score(y_test, y_pred_log_before)
recall_log_before = recall_score(y_test, y_pred_log_before)

st.write("Akurasi Logistic Regression Sebelum Normalisasi:", accuracy_log_before)
st.write("F1 Score Logistic Regression Sebelum Normalisasi:", f1_log_before)
st.write("Precision Logistic Regression Sebelum Normalisasi:", precision_log_before)
st.write("Recall Logistic Regression Sebelum Normalisasi:", recall_log_before)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf_before = rf_model.predict(X_test)
accuracy_rf_before = accuracy_score(y_test, y_pred_rf_before)
f1_rf_before = f1_score(y_test, y_pred_rf_before)
precision_rf_before = precision_score(y_test, y_pred_rf_before)
recall_rf_before = recall_score(y_test, y_pred_rf_before)

st.write("Akurasi Random Forest Sebelum Normalisasi:", accuracy_rf_before)
st.write("F1 Score Random Forest Sebelum Normalisasi:", f1_rf_before)
st.write("Precision Random Forest Sebelum Normalisasi:", precision_rf_before)
st.write("Recall Random Forest Sebelum Normalisasi:", recall_rf_before)

# SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm_before = svm_model.predict(X_test)
accuracy_svm_before = accuracy_score(y_test, y_pred_svm_before)
f1_svm_before = f1_score(y_test, y_pred_svm_before)
precision_svm_before = precision_score(y_test, y_pred_svm_before)
recall_svm_before = recall_score(y_test, y_pred_svm_before)

st.write("Akurasi SVM Sebelum Normalisasi:", accuracy_svm_before)
st.write("F1 Score SVM Sebelum Normalisasi:", f1_svm_before)
st.write("Precision SVM Sebelum Normalisasi:", precision_svm_before)
st.write("Recall SVM Sebelum Normalisasi:", recall_svm_before)

# Confusion Matrix
st.subheader('Confusion Matrix')
fig, ax = plt.subplots(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log_before), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(2, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf_before), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(2, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_svm_before), annot=True, fmt='d', cmap='Blues')
plt.title('SVM')
plt.ylabel('Actual')
plt.xlabel('Predicted')

st.pyplot(fig)

### 7. Normalisasi Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### 8. Model Setelah Normalisasi
st.subheader("Evaluasi Setelah Normalisasi")

# Logistic Regression
log_model.fit(X_train, y_train)
y_pred_log_after = log_model.predict(X_test)
accuracy_log_after = accuracy_score(y_test, y_pred_log_after)
f1_log_after = f1_score(y_test, y_pred_log_after)
precision_log_after = precision_score(y_test, y_pred_log_after)
recall_log_after = recall_score(y_test, y_pred_log_after)

st.write("Akurasi Logistic Regression Setelah Normalisasi:", accuracy_log_after)
st.write("F1 Score Logistic Regression Setelah Normalisasi:", f1_log_after)
st.write("Precision Logistic Regression Setelah Normalisasi:", precision_log_after)
st.write("Recall Logistic Regression Setelah Normalisasi:", recall_log_after)

# Random Forest
rf_model.fit(X_train, y_train)
y_pred_rf_after = rf_model.predict(X_test)
accuracy_rf_after = accuracy_score(y_test, y_pred_rf_after)
f1_rf_after = f1_score(y_test, y_pred_rf_after)
precision_rf_after = precision_score(y_test, y_pred_rf_after)
recall_rf_after = recall_score(y_test, y_pred_rf_after)

st.write("Akurasi Random Forest Setelah Normalisasi:", accuracy_rf_after)
st.write("F1 Score Random Forest Setelah Normalisasi:", f1_rf_after)
st.write("Precision Random Forest Setelah Normalisasi:", precision_rf_after)
st.write("Recall Random Forest Setelah Normalisasi:", recall_rf_after)

# SVM
svm_model.fit(X_train, y_train)
y_pred_svm_after = svm_model.predict(X_test)
accuracy_svm_after = accuracy_score(y_test, y_pred_svm_after)
f1_svm_after = f1_score(y_test, y_pred_svm_after)
precision_svm_after = precision_score(y_test, y_pred_svm_after)
recall_svm_after = recall_score(y_test, y_pred_svm_after)

st.write("Akurasi SVM Setelah Normalisasi:", accuracy_svm_after)
st.write("F1 Score SVM Setelah Normalisasi:", f1_svm_after)
st.write("Precision SVM Setelah Normalisasi:", precision_svm_after)
st.write("Recall SVM Setelah Normalisasi:", recall_svm_after)

### 9. Visualisasi Confusion Matrix
st.subheader('Confusion Matrix')

fig, ax = plt.subplots(figsize=(15, 10))
plt.subplot(2, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log_before), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression (Sebelum Normalisasi)')

plt.subplot(2, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf_before), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest (Sebelum Normalisasi)')

plt.subplot(2, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_svm_before), annot=True, fmt='d', cmap='Blues')
plt.title('SVM (Sebelum Normalisasi)')

plt.subplot(2, 3, 4)
sns.heatmap(confusion_matrix(y_test, y_pred_log_after), annot=True, fmt='d', cmap='Greens')
plt.title('Logistic Regression (Setelah Normalisasi)')

plt.subplot(2, 3, 5)
sns.heatmap(confusion_matrix(y_test, y_pred_rf_after), annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest (Setelah Normalisasi)')

plt.subplot(2, 3, 6)
sns.heatmap(confusion_matrix(y_test, y_pred_svm_after), annot=True, fmt='d', cmap='Greens')
plt.title('SVM (Setelah Normalisasi)')

st.pyplot(fig)

### 10. Cross-Validation untuk Semua Model
st.subheader("Cross-Validation Scores")

# Logistic Regression
log_cv_scores = cross_val_score(log_model, X, y, cv=5)
st.write("Logistic Regression CV Score:", log_cv_scores.mean())

# Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
st.write("Random Forest CV Score:", rf_cv_scores.mean())

# SVM
svm_cv_scores = cross_val_score(svm_model, X, y, cv=5)
st.write("SVM CV Score:", svm_cv_scores.mean())