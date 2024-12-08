import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Memuat dataset
try:
    data = pd.read_csv('iranian_churn_clean.csv')  # Ganti dengan path file CSV Anda
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print("File 'iranian_churn_clean.csv' tidak ditemukan. Pastikan path file sudah benar.")
    exit()

# Menentukan fitur dan target (sesuaikan dengan nama kolom dalam dataset Anda)
features = [
    "Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age", 
    "Customer Value"
]  # Sesuaikan dengan kolom fitur
target = "Churn"  # Sesuaikan dengan nama kolom target dalam dataset

# Memastikan semua kolom fitur dan target ada dalam dataset
missing_columns = [col for col in features + [target] if col not in data.columns]
if missing_columns:
    print(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")
    exit()

# Membagi data menjadi fitur dan target
X = data[features]
y = data[target]

# Mengatasi missing values
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("Dataset memiliki missing values. Missing values akan diisi dengan nilai median (untuk numerik) atau modus (untuk kategorikal).")
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0])  # Isi modus untuk kategorikal
        else:
            X[col] = X[col].fillna(X[col].median())  # Isi median untuk numerik
    y = y.fillna(y.mode()[0])  # Isi target dengan modus

# Jika ada kolom kategorikal, ubah menjadi numerik
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Melatih model DecisionTreeClassifier (CART)
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Memprediksi dan mengevaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')

# Menyimpan model ke file
model_filename = 'Cart-Customer_Churn.pkl'
joblib.dump(model, model_filename)
print(f"Model disimpan ke file: {model_filename}")
