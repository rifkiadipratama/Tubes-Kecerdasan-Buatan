import pandas as pd
import pyodbc

# Koneksi ke SQL Server
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=LAPTOP-4Q9VNCT5;'  # Ganti sesuai nama servermu
    'DATABASE=tubesAI;'         # Ganti sesuai nama databasenya
    'UID=sa;'                   # Ganti sesuai user
    'PWD=Rifkiap18'             # Ganti sesuai password
)

# Ambil data dari tabel tubes
df = pd.read_sql("""
    SELECT 
        arus_listrik, temperature, vibration, 
        label, label_downtime 
    FROM tubes
""", conn)

# Pisahkan fitur dan target
X = df[['arus_listrik', 'temperature', 'vibration']]  # Fitur

# Target 1: untuk predictive maintenance
y_maintenance = df['label']

# Target 2: untuk estimasi downtime
y_downtime = df['label_downtime']

# Tampilkan 5 data teratas untuk memastikan
print(df.head())
