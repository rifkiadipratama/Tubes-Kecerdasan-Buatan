import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

def train_models():
    # Koneksi ke SQL Server
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=LAPTOP-4Q9VNCT5;'
        'DATABASE=tubesAI;'
        'UID=sa;'
        'PWD=Rifkiap18'
    )

    # Ambil data dari SQL Server
    df = pd.read_sql("SELECT * FROM tubes WHERE label IS NOT NULL AND label_downtime IS NOT NULL", conn)

    # Pastikan data tidak kosong
    if df.empty:
        print("Tidak ada data untuk pelatihan model.")
        conn.close()
        return

    # Pisahkan fitur dan target untuk predictive maintenance
    X = df[['temperature', 'vibration', 'arus_listrik']]  # Fitur
    y_class = df['label']  # Target untuk klasifikasi (kondisi motor)

    # Pisahkan fitur dan target untuk downtime motor
    y_reg = df['label_downtime']  # Target untuk regresi (downtime motor)

    # Membagi data menjadi train dan test
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    # ------------------------- Random Forest Classifier (Predictive Maintenance) -------------------------
    # Membuat dan melatih model Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train_class)

    # Prediksi menggunakan model yang telah dilatih
    y_pred_class = rf_classifier.predict(X_test)

    # Hitung akurasi
    accuracy = accuracy_score(y_test_class, y_pred_class)
    print(f"Akurasi Random Forest Classifier untuk Predictive Maintenance: {accuracy * 100:.2f}%")

    # ------------------------- Random Forest Regressor (Downtime Motor) -------------------------
    # Membuat dan melatih model Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train_reg)

    # Prediksi menggunakan model yang telah dilatih
    y_pred_reg = rf_regressor.predict(X_test)

    # Hitung error regresi
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    print(f"Mean Absolute Error (MAE) Random Forest Regressor untuk Downtime Motor: {mae:.2f} jam")

    # Simpan model jika diperlukan (opsional)
    import joblib
    joblib.dump(rf_classifier, 'rf_classifier_model.joblib')
    joblib.dump(rf_regressor, 'rf_regressor_model.joblib')

    conn.close()

if __name__ == "__main__":
    train_models()
