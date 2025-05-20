import pyodbc
import pandas as pd
import joblib

# Load model
clf = joblib.load("rf_classifier_model.joblib")  # Classifier
reg = joblib.load("rf_regressor_model.joblib")  # Regressor

# Koneksi ke database
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=LAPTOP-4Q9VNCT5;'
    'DATABASE=tubesAI;'
    'UID=sa;'
    'PWD=Rifkiap18'
)
cursor = conn.cursor()

# Ambil data tanpa label
df = pd.read_sql("SELECT * FROM tubes WHERE label IS NULL OR label_downtime IS NULL", conn)

if not df.empty:
    # Ambil fitur input
    X = df[['temperature', 'vibration', 'arus_listrik']]

    # Prediksi label dan downtime
    pred_labels = clf.predict(X)
    pred_downtime = reg.predict(X)

    # Update hasil ke SQL Server
    for i in range(len(df)):
        cursor.execute(
            "UPDATE tubes SET label = ?, label_downtime = ? WHERE ID = ?",
            str(pred_labels[i]),
            float(pred_downtime[i]),
            int(df.loc[i, 'ID'])
    )

    conn.commit()
    print("Prediksi berhasil ditambahkan ke SQL Server.")
else:
    print("Tidak ada data baru untuk diprediksi.")

conn.close()
