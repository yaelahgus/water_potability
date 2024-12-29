import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. Pengumpulan Data (Membaca dataset)
water_data = pd.read_csv('data/water_potability.csv')
water_data.head()

# 2. Menelaah Data
st.write("Informasi Dataset:")
st.write(water_data.info())
st.write("\nJumlah Baris dan Kolom:", water_data.shape)
st.write("\nLima Baris Pertama:")
st.write(water_data.head())
st.write("\nJumlah Nilai Unik Tiap Kolom:")
st.write(water_data.nunique())

# 3. Validasi Data
st.write("\nCek Missing Values:")
st.write(water_data.isnull().sum())

# Imputasi Missing Values dengan Mean
water_data.fillna(water_data.mean(), inplace=True)

# Visualisasi Distribusi Data
st.subheader('Distribusi Kelayakan Minum (Potability)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='Potability', data=water_data, ax=ax)
st.pyplot(fig)

# Korelasi Heatmap
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

# 4. Pemodelan
X = water_data.drop('Potability', axis=1)
y = water_data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluasi Sebelum Normalisasi
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy_before_norm = accuracy_score(y_test, y_pred_log)
st.write("Akurasi Logistic Regression Sebelum Normalisasi:", accuracy_before_norm)

# Normalisasi Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Setelah Normalisasi
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy_after_norm = accuracy_score(y_test, y_pred_log)
st.write("Akurasi Logistic Regression Setelah Normalisasi:", accuracy_after_norm)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
st.subheader("Random Forest Model Evaluation")
st.write(classification_report(y_test, y_pred_rf, zero_division=1))

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
st.subheader("SVM Model Evaluation")
st.write(classification_report(y_test, y_pred_svm, zero_division=1))

# Confusion Matrix
st.subheader('Confusion Matrix')
fig, ax = plt.subplots(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('SVM')
plt.ylabel('Actual')
plt.xlabel('Predicted')

st.pyplot(fig)

# Feature Importance (Random Forest)
st.subheader('Feature Importance - Random Forest')
fig, ax = plt.subplots(figsize=(10, 6))
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title('Feature Importance - Random Forest')
st.pyplot(fig)

# Cross-Validation
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
st.write("Random Forest CV Score:", rf_cv_scores.mean())

# ROC Curve
st.subheader('ROC Curve')
fig, ax = plt.subplots(figsize=(10, 6))
y_score_rf = rf_model.predict_proba(X_test)[:, 1]
y_score_svm = svm_model.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest AUC = {roc_auc_rf:.2f}')
plt.plot(fpr_svm, tpr_svm, label=f'SVM AUC = {roc_auc_svm:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
st.pyplot(fig)

# Deployment dengan Streamlit
st.title('Klasifikasi Kelayakan Air Minum')

uploaded_file = st.file_uploader("Upload file CSV untuk prediksi")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.fillna(data.mean(), inplace=True)
    data_scaled = scaler.transform(data)
    predictions = rf_model.predict(data_scaled)
    st.write("Prediksi Kelayakan Air Minum:", predictions)

st.write("Akurasi Logistic Regression Setelah Normalisasi:", accuracy_after_norm)
st.write("Akurasi Logistic Regression Sebelum Normalisasi:", accuracy_before_norm)
st.write("Random Forest CV Score:", rf_cv_scores.mean())
