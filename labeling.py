import pyodbc
import pandas as pd

def label_downtime(row):
    """
    Fungsi untuk memberi label downtime motor berdasarkan fitur yang ada.
    Estimasi downtime berdasarkan nilai dari parameter (misalnya temperature, vibration, arus_listrik).
    """
    # Mendefinisikan aturan downtime berdasarkan kondisi motor
    downtime = None

    # Jika motor dalam kondisi rusak atau hampir rusak
    if row['temperature'] >= 60 or row['vibration'] >= 0.3:
        downtime = 0  # Waktu downtime sangat dekat, motor harus segera dihentikan
    elif row['temperature'] >= 55 or row['vibration'] >= 0.2:
        downtime = 12  # Perkiraan downtime 12 jam
    elif row['temperature'] >= 50 or row['vibration'] >= 0.15:
        downtime = 24  # Perkiraan downtime 24 jam
    else:
        downtime = 48  # Motor normal, downtime lebih lama, estimasi 48 jam (2 hari)
    
    return downtime

def run_labeling():
    # Koneksi ke SQL Server
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=LAPTOP-4Q9VNCT5;'
        'DATABASE=tubesAI;'
        'UID=sa;'
        'PWD=Rifkiap18'
    )

    # Ambil data dari SQL Server
    df = pd.read_sql("SELECT * FROM tubes", conn)

    # Periksa apakah data kosong
    if df.empty:
        print("Tidak ada data untuk dilabeli.")
        conn.close()
        return

    # Fungsi labeling untuk predictive maintenance
    def label_row(row):
        if row['temperature'] >= 60 or row['vibration'] >= 0.3:
            return 'Rusak'  # Kondisi kritis, harus dihentikan
        elif row['temperature'] >= 55 or row['vibration'] >= 0.2:
            return 'Perlu Perawatan'  # Kondisi mulai memburuk
        elif row['temperature'] >= 50 or row['vibration'] >= 0.15:
            return 'Cek Berkala'  # Cek berkala, kondisi mulai menunjukkan tanda-tanda
        else:
            return 'Normal'  # Kondisi aman

    # Tambah kolom label untuk predictive maintenance
    df['label'] = df.apply(label_row, axis=1)

    # Tambah kolom label_downtime untuk downtime motor
    df['label_downtime'] = df.apply(label_downtime, axis=1)

    # Update label ke SQL Server
    cursor = conn.cursor()
    for index, row in df.iterrows():
        cursor.execute(
            "UPDATE tubes SET label = ?, label_downtime = ? WHERE ID = ?",
            row['label'], row['label_downtime'], row['ID']
        )
    conn.commit()
    conn.close()

    print("Label dan downtime berhasil diperbarui.")

if __name__ == "__main__":
    run_labeling()
