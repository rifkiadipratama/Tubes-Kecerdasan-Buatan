import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QWidget, QGroupBox, QCheckBox, QComboBox, QLineEdit, QMessageBox, QScrollArea,
    QDialog, QDialogButtonBox, QTextEdit, QTableWidgetItem, QFileDialog, QTableWidget, QSlider, QTabWidget, QStackedWidget
)
from PyQt6.QtCore import QStandardPaths, Qt
from PyQt6.QtWidgets import QDialogButtonBox, QFrame, QFileDialog, QMessageBox 
from sklearn.preprocessing import LabelEncoder
from PyQt5.QtGui import QTextCursor
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtCore import Qt
import traceback
from typing import Union
from sklearn.exceptions import NotFittedError
import serial
import serial.tools.list_ports
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
from PyQt6 import QtWidgets
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import tkinter as tk
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QGridLayout
from PyQt6.QtGui import QFont, QPixmap, QMouseEvent
import pyodbc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime
import csv
import pandas as pd
from joblib import dump  # Tambahkan ini di bagian import atas file Anda
import joblib
from pathlib import Path
import glob 

#-----training------#
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

###----evaluation-----###
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox 

import os
import joblib
from joblib import load  # Import the load function

import random

class EmittingStream(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):  # Diperlukan untuk kompatibilitas dengan sys.stdout
        pass

class ClickableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked()
        super().mousePressEvent(event)
    def clicked(self):
        # This method is intended to be overridden.
        pass

class MainWindow(QtWidgets.QMainWindow, tk.Tk):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Predictive Maintenance with Machine Learning")
        self.setGeometry(100, 100, 1180, 730)
        
        self.load_latest_models()

        self.class_mapping = {
            0: {"label": "getaran bahaya", "color": "red", "description": "Getaran melebihi ambang batas aman"},
            1: {"label": "indikasi rusak", "color": "#FF8C00", "description": "Indikasi awal kerusakan terdeteksi"},
            2: {"label": "perlu perawatan", "color": "orange", "description": "Perlu dilakukan pemeriksaan/perawatan"},
            3: {"label": "suhu tinggi, getaran rendah", "color": "#FF6347", "description": "Suhu tinggi namun getaran masih aman"},
            4: {"label": "suhu rendah, getaran tinggi", "color": "#1E90FF", "description": "Suhu normal namun getaran cukup tinggi"},
            5: {"label": "normal", "color": "green", "description": "Kondisi mesin normal"}
        }
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_data)
        self.is_reading = False  # Tambahkan flag untuk status pembacaan
        
        # Terminal log GUI
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)

        self.classifier_frame = None  # Inisialisasi
        self.regressor_frame = None

         # Initialize terminal_output as a list (or any other suitable data structure)
        self.terminal_output = []

        self.vibration_widgets = []  # Menyimpan widget di menu vibration
        self.training_widgets = []

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0, 0, 0)
        self.main_layout.setSpacing(10)

        # Jangan dimasukkan ke layout
        self.title_label = QLabel("PREDICTIVE MAINTENANCE", self.central_widget)
        self.title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.title_label.resize(580, 50)
        self.title_label.move(425, 0)  # X, Y position
        self.title_label.show()

        # Notification icon label
        self.notification_icon = ClickableLabel(self)
        # Load the notification icon image from your computer directory
        pixmap = QPixmap("notification off.png")  # Replace with the exact path if needed
        pixmap = pixmap.scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.notification_icon.setPixmap(pixmap)
        
        # Position notification icon right aligned and vertically centered relative to title label
        icon_x = self.title_label.x() + self.title_label.width() + 100  # 10 px right offset
        icon_y = self.title_label.y() + (self.title_label.height() - 8)   # vertical center
        self.notification_icon.move(icon_x, icon_y)
        self.notification_icon.resize(24, 24)
        self.notification_icon.show()

        # Connect click signal to handler
        self.notification_icon.clicked = self.show_notification_window

        # Section title
        self.section_title = QLabel("")
        self.section_title.setFont(QFont("Arial", 11))
        self.section_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.main_layout.addWidget(self.section_title)

        # Container untuk isi dinamis
        self.content_area = QVBoxLayout()
        self.main_layout.addLayout(self.content_area)

        # Menu bar
        menu_bar = self.menuBar()
        menu_bar.addAction(QAction("Dashboard", self, triggered=self.show_vibration))
        menu_bar.addAction(QAction("Training", self, triggered= self.show_training))
        menu_bar.addAction(QAction("Evaluasi", self, triggered=self.show_evaluation))
        menu_bar.addAction(QAction("Database", self, triggered=self.show_database))
        
        # Layout indikator (hanya di vibration)
        self.indicator_layout = None
        
        self.show_vibration()
        # # Connect to database when the program starts
        # self.connect_to_database()

    def set_notification_icon(self, icon_path: str):
        pixmap = QPixmap(icon_path)
        pixmap = pixmap.scaled(20, 20, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.notification_icon.setPixmap(pixmap)

    def connect_to_db(self):
        try:
            conn = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=LAPTOP-4Q9VNCT5;'  # Ganti sesuai nama servermu
                'DATABASE=TubesAI;'         # Ganti sesuai nama databasenya
                'UID=sa;'                   # Ganti sesuai user
                'PWD=Rifkiap18'             # Ganti sesuai password
            )
            return conn
        except Exception as e:
            print(f"Connection error: {e}")
            return None

    def connect_to_database(self):
        try:
            conn = pyodbc.connect(
                'DRIVER={SQL Server};'
                'SERVER=LAPTOP-4Q9VNCT5;'  # Ganti sesuai nama servermu
                'DATABASE=TubesAI;'         # Ganti sesuai nama databasenya
                'UID=sa;'                   # Ganti sesuai user
                'PWD=Rifkiap18'             # Ganti sesuai password
            )
            cursor = conn.cursor()
            
            # Dapatkan nama database aktif
            cursor.execute("SELECT DB_NAME()")  # Tanpa parameter
            db_name = cursor.fetchone()[0]
           
            # Ambil nama tabel pertama dari database
            cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            result = cursor.fetchone()
            
            if result:
                table_name = result[0]
                
                # Hitung jumlah baris dari tabel tersebut
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                # Update label
                self.db_label1.setText(f"Database: {db_name}")
                self.db_label2.setText(f"Tabel: {table_name}")
                self.db_label3.setText(f"Jumlah Baris: {row_count}")
                self.connection_status_label.setText("✅ Berhasil Connect")
                self.connection_status_label.setStyleSheet("color: green")
            else:
                self.db_label1.setText(db_name)
                self.db_label2.setText("Tidak ada tabel.")
                self.connection_status_label.setText("Gagal Connect")
                self.db_label3.setText("")

            cursor.close()
            
            conn.close()

        except pyodbc.Error as e:
            self.db_label1.setText("Database: -")
            self.db_label2.setText("Tabel: -")
            self.db_label3.setText("Jumlah Baris: -")
            self.connection_status_label.setText("❌ Gagal Connect")
            self.connection_status_label.setStyleSheet("color: red")
            print("❌ Gagal koneksi ke database.")
            print("Error:", e)

    # def append_to_terminal(self, text):
    #     self.terminal_output.append(text)
    #     self.terminal_output.moveCursor(QTextCursor.MoveOperation.End)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def clear_content(self):
        self.clear_layout(self.content_area)
        if self.indicator_layout:
            self.clear_layout(self.indicator_layout)
            self.main_layout.removeItem(self.indicator_layout)
            self.indicator_layout = None

        # Hapus semua widget di vibration_widgets
        if hasattr(self, 'vibration_widgets'):
            for widget in self.vibration_widgets:
                widget.setParent(None)
            self.vibration_widgets = []

        # Hapus semua widget di training_widgets
        if hasattr(self, 'training_widgets'):
            for widget in self.training_widgets:
                widget.setParent(None)
            self.training_widgets = []

        # Hapus semua widget di evaluation_widgets
        if hasattr(self, 'evaluation_widgets'):
            for widget in self.evaluation_widgets:
                widget.setParent(None)
            self.evaluation_widgets = []

        if hasattr(self, 'database_widgets'):
            for widget in self.database_widgets:
                widget.setParent(None)
            self.database_widgets = []

    def show_evaluation(self):
        self.clear_content()
        self.evaluation_widgets = []
        
        self.evaluation_label =QLabel("EVALUATION SECTION", self)
        # Manually position the label at x=50, y=30
        self.evaluation_label.move(20, 70)
        # Optionally you can set fixed size or font to adjust appearance
        self.evaluation_label.resize(200, 40)  # width, height
        font = self.evaluation_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.evaluation_label.setFont(font)
        self.evaluation_label.show()

        # Container widget untuk layout horizontal
        self.eval_container = QWidget(self)
        self.eval_container.move(20, 90)  # posisi container
        self.eval_container.resize(1000, 50)  # ukuran container

        # Layout horizontal
        eval_layout = QHBoxLayout(self.eval_container)
        eval_layout.setContentsMargins(10, 0, 10, 0)
        eval_layout.setSpacing(15)  # jarak antar widget (15 px)

        # Indikator model
        self.model_label = QLabel("Model: Random Forest", self.eval_container)

        # Checkbox multi-pilihan
        self.classifier_checkbox = QCheckBox("Klasifikasi", self.eval_container)
        self.regressor_checkbox = QCheckBox("Regresi", self.eval_container)

        # Tombol evaluasi
        self.evaluate_button = QPushButton("Evaluasi Model", self.eval_container)
        self.evaluate_button.setStyleSheet("background-color: #077A7D")
        self.evaluate_button.clicked.connect(self.evaluate_model_with_prompt)

        # Tambahkan widget ke layout horizontal
        eval_layout.addWidget(self.model_label)
        eval_layout.addWidget(self.classifier_checkbox)
        eval_layout.addWidget(self.regressor_checkbox)
        eval_layout.addWidget(self.evaluate_button)
       
        # Terapkan layout ke container dan tampilkan
        self.eval_container.setLayout(eval_layout)
        self.eval_container.show()

        # Buat GroupBox sebagai frame evaluasi
        self.classifier_frame = QGroupBox("Hasil Evaluasi Klasifikasi", self)
        self.classifier_frame.move(20, 130)
        self.classifier_frame.resize(300, 180)

        # Layout vertikal di dalam GroupBox
        classifier_layout = QVBoxLayout(self.classifier_frame)

        # Label hasil evaluasi (default)
        self.accuracy_label = QLabel("Akurasi: -")
        self.precision_label = QLabel("Precision: -")
        self.recall_label = QLabel("Recall: -")
        self.f1_label = QLabel("F1-Score: -")
        self.cm_values = QLabel("Confusion Matrix:\nTP: -  FN: -\nFP: -  TN: -")

        # Tambahkan label ke layout
        classifier_layout.addWidget(self.accuracy_label)
        classifier_layout.addWidget(self.precision_label)
        classifier_layout.addWidget(self.recall_label)
        classifier_layout.addWidget(self.f1_label)
        classifier_layout.addWidget(self.cm_values)

        self.classifier_frame.setLayout(classifier_layout)
        self.classifier_frame.show()

        # # Masukkan ke evaluation_widgets agar bisa dihapus nanti
        # self.evaluation_widgets.append(self.classifier_frame)

        # === Frame Evaluasi untuk Regresi ===
        self.regressor_frame = QGroupBox("Hasil Evaluasi Regresi", self)
        self.regressor_frame.move(340, 130)  # Letakkan di samping frame klasifikasi
        self.regressor_frame.resize(300, 160)

        regressor_layout = QVBoxLayout(self.regressor_frame)

        # Label evaluasi regresi (default)
        self.mae_label = QLabel("MAE: -")
        self.mse_label = QLabel("MSE: -")
        self.rmse_label = QLabel("RMSE: -")
        self.r2_label = QLabel("R² Score: -")

        regressor_layout.addWidget(self.mae_label)
        regressor_layout.addWidget(self.mse_label)
        regressor_layout.addWidget(self.rmse_label)
        regressor_layout.addWidget(self.r2_label)

        self.regressor_frame.setLayout(regressor_layout)
        self.regressor_frame.show()

        # === Container Grafik ===
        self.graph_container = QFrame(self)
        self.graph_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.graph_container.setGeometry(20, 320, 1100, 370)  # x, y, width, height
        self.graph_container.show()

        # === Grafik Confusion Matrix ===
        self.cm_fig, self.cm_ax = plt.subplots(figsize=(12,6))
        self.cm_canvas = FigureCanvas(self.cm_fig)
        self.cm_canvas.setParent(self.graph_container)
        self.cm_canvas.move(20, 10)
        self.cm_canvas.resize(400, 350)

        # Dummy 3x3 confusion matrix
        cm = np.array([
            [50, 2, 3],
            [4, 45, 1],
            [2, 3, 48]
        ])
        self.cm_ax.clear()
        self.cm_ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        self.cm_ax.set_title("Confusion Matrix")
        self.cm_ax.set_xlabel("Predicted")
        self.cm_ax.set_ylabel("Actual")

        # Label sumbu
        num_classes = cm.shape[0]
        self.cm_ax.set_xticks(np.arange(num_classes))
        self.cm_ax.set_yticks(np.arange(num_classes))
        self.cm_ax.set_xticklabels([f"Class {i}" for i in range(num_classes)])
        self.cm_ax.set_yticklabels([f"Class {i}" for i in range(num_classes)])

        # Tambahkan angka ke dalam cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                color = "white" if value > cm.max() / 2 else "black"
                self.cm_ax.text(j, i, str(value), ha='center', va='center', color=color)

        self.cm_canvas.draw()
        self.cm_canvas.show()

        # === Grafik Scatter Plot ===
        self.reg_fig, self.reg_ax = plt.subplots(figsize=(12, 6))
        self.reg_canvas = FigureCanvas(self.reg_fig)
        self.reg_canvas.setParent(self.graph_container)
        self.reg_canvas.move(480, 10)
        self.reg_canvas.resize(400, 350)

        # Dummy data regresi
        y_test = np.array([3.0, 4.5, 5.0, 6.2, 7.1])
        y_pred = np.array([2.8, 4.7, 5.1, 6.0, 7.0])

        self.reg_ax.clear()
        self.reg_ax.scatter(y_test, y_pred, c='blue')
        self.reg_ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        self.reg_ax.set_title("Scatter Plot Regressor")
        self.reg_ax.set_xlabel("Actual")
        self.reg_ax.set_ylabel("Predicted")
        self.reg_canvas.draw()
        self.reg_canvas.show()

        # === Terminal Monitor ===
        self.terminal_monitor2 = QWidget(self)
        self.terminal_monitor2.setGeometry(650, 110, 490, 180)  # Letaknya di bawah model_container

        terminal_layout2 = QVBoxLayout()

        # Tombol Clear di pojok kanan atas
        clear_layout2 = QHBoxLayout()
        clear_layout2.addStretch()  # Push tombol ke kanan
        self.clear_button2 = QPushButton("Clear")
        self.clear_button2.setFixedSize(60, 25)
        self.clear_button2.clicked.connect(self.clear_terminal_output2)
        clear_layout2.addWidget(self.clear_button2)

        # TextEdit untuk log output
        self.terminal_output2 = QTextEdit()
        self.terminal_output2.setReadOnly(True)
        self.terminal_output2.setStyleSheet("background-color: black; color: lime; font-family: Consolas;")

        terminal_layout2.addLayout(clear_layout2)
        terminal_layout2.addWidget(self.terminal_output2)
        self.terminal_monitor2.setLayout(terminal_layout2)
        self.terminal_monitor2.show()

        self.evaluation_widgets.append(self.terminal_monitor2)

        # Tambahkan semua widget ini ke evaluation_widgets agar bisa dibersihkan nanti
        self.evaluation_widgets = [
            self.evaluation_label,
            self.eval_container,
            self.classifier_frame,
            self.regressor_frame,
            self.terminal_monitor2,
            self.graph_container
        ]
    
    def evaluate_model(self):
        """Melakukan evaluasi model dengan metrik yang sesuai dan visualisasi"""
        try:
            # ===== 1. SETUP AWAL =====
            self.set_notification_icon("notification off.png")
            self.terminal_output2.append("Memulai proses evaluasi model...")
            
            # ===== 2. VALIDASI MODEL TERLATIH =====
            if not hasattr(self, 'classifier_model') and not hasattr(self, 'regressor_model'):
                raise ValueError("Tidak ada model yang telah dilatih!")
            
            # Validasi data test eksis
            if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
                raise ValueError("Data test belum dimuat untuk evaluasi!")
            
            # ===== 3. PREPROCESSING DATA TEST =====
            # Label encoding untuk kolom kategorikal (gunakan encoder yang sama seperti training)
            if hasattr(self, 'X_train'):
                categorical_cols = self.X_train.select_dtypes(include=['object']).columns
                for column in categorical_cols:
                    if hasattr(self, 'label_encoders') and column in self.label_encoders:
                        le = self.label_encoders[column]
                        self.X_test[column] = le.transform(self.X_test[column].astype(str))
            
            # Konversi ke numpy array untuk konsistensi
            X_test = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
            
            # Scaling data (gunakan scaler yang sama seperti training)
            if hasattr(self, 'scaler') and self.scaler is not None:
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # ===== 4. EVALUASI CLASSIFIER =====
            if hasattr(self, 'classifier_model'):
                from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                        f1_score, confusion_matrix, classification_report,
                                        roc_curve, auc, precision_recall_curve)
                import matplotlib.pyplot as plt
                from sklearn.preprocessing import label_binarize
                
                # Prediksi pada data test
                y_pred = self.classifier_model.predict(X_test_scaled)
                y_proba = self.classifier_model.predict_proba(X_test_scaled)
                
                # Hitung metrik evaluasi
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')
                recall = recall_score(self.y_test, y_pred, average='weighted')
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Tampilkan hasil di terminal
                self.terminal_output2.append("\n=== EVALUASI CLASSIFIER ===")
                self.terminal_output2.append(f"Akurasi: {accuracy:.4f}")
                self.terminal_output2.append(f"Presisi: {precision:.4f}")
                self.terminal_output2.append(f"Recall: {recall:.4f}")
                self.terminal_output2.append(f"F1-Score: {f1:.4f}")
                
                # Laporan klasifikasi lengkap
                self.terminal_output2.append("\nLaporan Klasifikasi:")
                report = classification_report(self.y_test, y_pred)
                self.terminal_output2.append(report)
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred)
                self.terminal_output2.append("\nConfusion Matrix:")
                self.terminal_output2.append(str(cm))
                
                # Visualisasi
                try:
                    # Confusion Matrix Plot
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.tight_layout()
                    plt.savefig('confusion_matrix.png')
                    plt.close()
                    
                    # ROC Curve (untuk klasifikasi biner)
                    if len(np.unique(self.y_test)) == 2:
                        fpr, tpr, _ = roc_curve(self.y_test, y_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
                        plt.figure()
                        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc="lower right")
                        plt.tight_layout()
                        plt.savefig('roc_curve.png')
                        plt.close()
                    
                    # Feature Importance
                    if hasattr(self.classifier_model, 'feature_importances_'):
                        importances = self.classifier_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        plt.figure(figsize=(10, 6))
                        plt.title("Feature Importances")
                        plt.bar(range(X_test.shape[1]), 
                                importances[indices],
                                color="r", align="center")
                        plt.xticks(range(X_test.shape[1]), 
                                [self.feature_names[i] for i in indices], 
                                rotation=90)
                        plt.xlim([-1, X_test.shape[1]])
                        plt.tight_layout()
                        plt.savefig('feature_importance.png')
                        plt.close()
                    
                except Exception as e:
                    self.terminal_output2.append(f"Gagal membuat visualisasi: {str(e)}")
            
            # ===== 5. EVALUASI REGRESSOR =====
            if hasattr(self, 'regressor_model'):
                from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                                        r2_score, explained_variance_score)
                import matplotlib.pyplot as plt
                
                # Prediksi pada data test
                y_pred = self.regressor_model.predict(X_test_scaled)
                
                # Hitung metrik evaluasi
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                evs = explained_variance_score(self.y_test, y_pred)
                
                # Tampilkan hasil di terminal
                self.terminal_output2.append("\n=== EVALUASI REGRESSOR ===")
                self.terminal_output2.append(f"MSE: {mse:.4f}")
                self.terminal_output2.append(f"MAE: {mae:.4f}")
                self.terminal_output2.append(f"R2 Score: {r2:.4f}")
                self.terminal_output2.append(f"Explained Variance Score: {evs:.4f}")
                
                # Visualisasi
                try:
                    # Scatter plot actual vs predicted
                    plt.figure(figsize=(8, 6))
                    plt.scatter(self.y_test, y_pred, alpha=0.5)
                    plt.plot([self.y_test.min(), self.y_test.max()], 
                            [self.y_test.min(), self.y_test.max()], 
                            'k--', lw=2)
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title('Actual vs Predicted Values')
                    plt.tight_layout()
                    plt.savefig('actual_vs_predicted.png')
                    plt.close()
                    
                    # Residual plot
                    residuals = self.y_test - y_pred
                    plt.figure(figsize=(8, 6))
                    plt.scatter(y_pred, residuals, alpha=0.5)
                    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), 
                            colors='k', linestyles='dashed')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot')
                    plt.tight_layout()
                    plt.savefig('residual_plot.png')
                    plt.close()
                    
                    # Feature Importance
                    if hasattr(self.regressor_model, 'feature_importances_'):
                        importances = self.regressor_model.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        
                        plt.figure(figsize=(10, 6))
                        plt.title("Feature Importances")
                        plt.bar(range(X_test.shape[1]), 
                                importances[indices],
                                color="r", align="center")
                        plt.xticks(range(X_test.shape[1]), 
                                [self.feature_names[i] for i in indices], 
                                rotation=90)
                        plt.xlim([-1, X_test.shape[1]])
                        plt.tight_layout()
                        plt.savefig('feature_importance_regressor.png')
                        plt.close()
                    
                except Exception as e:
                    self.terminal_output2.append(f"Gagal membuat visualisasi regressor: {str(e)}")
            
            # ===== 6. SIMPAN HASIL EVALUASI =====
            self.model_evaluation_results = {
                'classifier': {
                    'accuracy': accuracy if hasattr(self, 'classifier_model') else None,
                    'precision': precision if hasattr(self, 'classifier_model') else None,
                    'recall': recall if hasattr(self, 'classifier_model') else None,
                    'f1': f1 if hasattr(self, 'classifier_model') else None,
                    'confusion_matrix': cm.tolist() if hasattr(self, 'classifier_model') else None,
                    'classification_report': report if hasattr(self, 'classifier_model') else None
                },
                'regressor': {
                    'mse': mse if hasattr(self, 'regressor_model') else None,
                    'mae': mae if hasattr(self, 'regressor_model') else None,
                    'r2': r2 if hasattr(self, 'regressor_model') else None,
                    'explained_variance': evs if hasattr(self, 'regressor_model') else None
                }
            }
            
            # ===== 7. UPDATE UI AKHIR =====
            self.set_notification_icon("notification on.png")
            QMessageBox.information(self, "Sukses", "Evaluasi model berhasil!")
            self.terminal_output2.append("Evaluasi model berhasil")
            
            # Update plot jika diperlukan
            if hasattr(self, 'update_evaluation_plots'):
                self.update_evaluation_plots()
        
        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", f"Parameter tidak valid:\n{str(ve)}")
            self.terminal_output2.append(f"Error: {str(ve)}")
        except RuntimeError as re:
            QMessageBox.critical(self, "Evaluation Error", f"Proses evaluasi gagal:\n{str(re)}")
            self.terminal_output2.append(f"Error: {str(re)}")
        except Exception as e:
            QMessageBox.critical(self, "System Error", f"Error tidak terduga:\n{str(e)}")
            self.terminal_output2.append(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def evaluate_model_with_prompt(self):
        """Menjalankan evaluasi model dan menampilkan hasilnya di UI"""
        try:
            # Validasi model yang dipilih
            if not self.classifier_checkbox.isChecked() and not self.regressor_checkbox.isChecked():
                QMessageBox.warning(self, "Peringatan", "Pilih minimal satu jenis model untuk dievaluasi!")
                return
            
            # Jalankan evaluasi model
            self.evaluate_model()
            
            # ===== UPDATE HASIL CLASSIFIER =====
            if hasattr(self, 'model_evaluation_results') and 'classifier' in self.model_evaluation_results:
                classifier_results = self.model_evaluation_results['classifier']
                
                if classifier_results['accuracy'] is not None:
                    # Update label teks
                    self.accuracy_label.setText(f"Akurasi: {classifier_results['accuracy']:.4f}")
                    self.precision_label.setText(f"Precision: {classifier_results['precision']:.4f}")
                    self.recall_label.setText(f"Recall: {classifier_results['recall']:.4f}")
                    self.f1_label.setText(f"F1-Score: {classifier_results['f1']:.4f}")
                    
                    # Update confusion matrix
                    if classifier_results['confusion_matrix']:
                        cm = np.array(classifier_results['confusion_matrix'])
                        if cm.shape == (2, 2):  # Binary classification
                            cm_text = f"Confusion Matrix:\nTP: {cm[0,0]}  FN: {cm[0,1]}\nFP: {cm[1,0]}  TN: {cm[1,1]}"
                        else:
                            cm_text = "Confusion Matrix:\n" + "\n".join([" ".join(map(str, row)) for row in cm])
                        self.cm_values.setText(cm_text)
                        
                        # Update confusion matrix plot
                        self.cm_ax.clear()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_ax)
                        self.cm_ax.set_title("Confusion Matrix")
                        self.cm_ax.set_xlabel("Predicted")
                        self.cm_ax.set_ylabel("Actual")
                        self.cm_canvas.draw()
                    
                    # Tampilkan frame classifier jika sebelumnya tersembunyi
                    self.classifier_frame.show()
            
            # ===== UPDATE HASIL REGRESSOR =====
            if hasattr(self, 'model_evaluation_results') and 'regressor' in self.model_evaluation_results:
                regressor_results = self.model_evaluation_results['regressor']
                
                if regressor_results['mse'] is not None:
                    # Update label teks
                    self.mae_label.setText(f"MAE: {regressor_results['mae']:.4f}")
                    self.mse_label.setText(f"MSE: {regressor_results['mse']:.4f}")
                    self.rmse_label.setText(f"RMSE: {np.sqrt(regressor_results['mse']):.4f}")
                    self.r2_label.setText(f"R² Score: {regressor_results['r2']:.4f}")
                    
                    # Update scatter plot
                    if hasattr(self, 'y_test') and hasattr(self, 'regressor_model'):
                        X_test_scaled = self.scaler.transform(self.X_test.values) if hasattr(self, 'scaler') else self.X_test.values
                        y_pred = self.regressor_model.predict(X_test_scaled)
                        
                        self.reg_ax.clear()
                        self.reg_ax.scatter(self.y_test, y_pred, c='blue')
                        self.reg_ax.plot([min(self.y_test), max(self.y_test)], 
                                        [min(self.y_test), max(self.y_test)], 'r--')
                        self.reg_ax.set_title("Scatter Plot Regressor")
                        self.reg_ax.set_xlabel("Actual")
                        self.reg_ax.set_ylabel("Predicted")
                        self.reg_canvas.draw()
                    
                    # Tampilkan frame regressor jika sebelumnya tersembunyi
                    self.regressor_frame.show()
            
            # Update terminal output
            if hasattr(self, 'terminal_output2'):
                self.terminal_output2.append("Evaluasi model selesai!")
                if hasattr(self, 'model_evaluation_results'):
                    self.terminal_output2.append(str(self.model_evaluation_results))
            
            QMessageBox.information(self, "Sukses", "Evaluasi model berhasil ditampilkan!")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menampilkan hasil evaluasi: {str(e)}")
            if hasattr(self, 'terminal_output2'):
                self.terminal_output2.append(f"Error: {str(e)}")

    def show_training(self):
        self.clear_content()
        
        # Training Label
        self.training_label = QLabel("TRAINING SECTION", self)
        self.training_label.move(20, 70)
        self.training_label.resize(200, 40)
        font = self.training_label.font()
        font.setPointSize(10)
        font.setBold(True)
        self.training_label.setFont(font)

        # Main Training Container
        self.training_container = QWidget(self)
        self.training_container.setGeometry(30, 100, 650, 180)
        container_layout = QHBoxLayout(self.training_container)

        # Model Container
        self.model_container = QWidget(self)
        self.model_container.setGeometry(30, 270, 650, 285)
        settingModel_layout = QVBoxLayout(self.model_container)

       # Terminal Monitor
        self.terminal_monitor = QWidget(self)
        self.terminal_monitor.setGeometry(30, 540, 1020, 200)  # Perlebar area agar indikator muat

        # Layout utama horizontal agar terminal dan indikator bisa sejajar
        main_terminal_layout = QHBoxLayout(self.terminal_monitor)

        # ----------- Bagian Kiri: Terminal Output + Clear Button -----------
        left_terminal_widget = QWidget()
        left_terminal_layout = QVBoxLayout(left_terminal_widget)

        # Tombol Clear
        clear_layout = QHBoxLayout()
        clear_layout.addStretch()
        self.clear_button = QPushButton("Clear")
        self.clear_button.setFixedSize(60, 25)
        self.clear_button.clicked.connect(self.clear_terminal_output)
        clear_layout.addWidget(self.clear_button)

        # Terminal Output
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setStyleSheet("background-color: black; color: lime; font-family: Consolas;")

        left_terminal_layout.addLayout(clear_layout)
        left_terminal_layout.addWidget(self.terminal_output)

        # ----------- Bagian Kanan: Indikator File Model -----------
        right_indicator_widget = QWidget()
        right_indicator_layout = QVBoxLayout(right_indicator_widget)

        self.classifier_label = QLabel("File hasil training model classifier: -")
        self.regressor_label = QLabel("File hasil training model regressor: -")

        right_indicator_layout.addWidget(self.classifier_label)
        right_indicator_layout.addWidget(self.regressor_label)
        right_indicator_layout.addStretch()  # Supaya label tetap di atas

        # Tambahkan ke layout utama
        main_terminal_layout.addWidget(left_terminal_widget, 3)
        main_terminal_layout.addWidget(right_indicator_widget, 2)

        # Training Box Model (Classifier)
        self.training_box_model = QGroupBox("Hasil Training")
        model_layout = QVBoxLayout(self.training_box_model)
        model_layout.addWidget(QLabel("Model: Random Forest Classifier"))
        model_layout.addWidget(QLabel("Akurasi: -%"))
        model_layout.addWidget(QLabel("Presisi: -%"))
        model_layout.addWidget(QLabel("Recall: -%"))
        model_layout.addWidget(QLabel("F1-Score: -%"))
        model_layout.addWidget(QLabel("Top Fitur: -"))

        # Training Box Model (Regressor)
        self.training_box_model_regressor = QGroupBox("Hasil Training")
        model2_layout = QVBoxLayout(self.training_box_model_regressor)
        model2_layout.addWidget(QLabel("Model: Random Forest Regressor"))
        model2_layout.addWidget(QLabel("MSE: -%"))
        model2_layout.addWidget(QLabel("MAE: -%"))
        model2_layout.addWidget(QLabel("RMSE: -%"))
        model2_layout.addWidget(QLabel("R^2 Score: -%"))
        model2_layout.addWidget(QLabel("Top Fitur: -"))

        # Training Box Database
        self.training_box_db = QGroupBox("Info Database")
        db_layout = QGridLayout(self.training_box_db)

        self.db_label1 = QLabel("Database: ", self)
        self.db_label2 = QLabel("Tabel: ", self)
        self.db_label3 = QLabel("Jumlah Baris: ", self)

        # Tambahkan ke layout (baris-baris utama)
        db_layout.addWidget(self.db_label1, 1, 0)
        db_layout.addWidget(self.db_label2, 2, 0)
        db_layout.addWidget(self.db_label3, 3, 0)

        self.connection_status_label = QLabel("✅ Berhasil Connect")
        self.connection_status_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        db_layout.addWidget(self.connection_status_label, 1, 1)

        # ========== CONTAINER KANAN ==========
        self.right_container = QWidget(self)
        self.right_container.setGeometry(680, 100, 480, 445)
        right_layout = QVBoxLayout(self.right_container)

        # Container utama untuk visualisasi
        self.graphics_box = QGroupBox("Visualisasi Model")
        graphics_layout = QVBoxLayout(self.graphics_box)

        # ===== TOMBOL SWITCH =====
        self.btn_container = QWidget()
        btn_layout = QHBoxLayout(self.btn_container)
        
        self.classifier_btn = QPushButton("Confusion Matrix")
        self.classifier_btn.setFixedSize(180, 35)
        self.classifier_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.classifier_btn.clicked.connect(self.show_classifier_graph)
        
        self.regressor_btn = QPushButton("Scatter Plot")
        self.regressor_btn.setFixedSize(180, 35)
        self.regressor_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #424242;
            }
        """)
        self.regressor_btn.clicked.connect(self.show_regressor_graph)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.classifier_btn)
        btn_layout.addWidget(self.regressor_btn)
        btn_layout.addStretch()

        # ===== AREA GRAFIK =====
        self.graph_stack = QStackedWidget()
        
        # Canvas Confusion Matrix
        self.cm_figure = Figure(figsize=(4.5, 3.5), dpi=100)
        self.cm_canvas = FigureCanvas(self.cm_figure)
        self.cm_toolbar = NavigationToolbar(self.cm_canvas, self)
        
        cm_container = QWidget()
        cm_layout = QVBoxLayout(cm_container)
        cm_layout.addWidget(self.cm_toolbar)
        cm_layout.addWidget(self.cm_canvas)
        self.graph_stack.addWidget(cm_container)
        
        # Canvas Scatter Plot
        self.sp_figure = Figure(figsize=(4.5, 3.5), dpi=100)
        self.sp_canvas = FigureCanvas(self.sp_figure)
        self.sp_toolbar = NavigationToolbar(self.sp_canvas, self)
        
        sp_container = QWidget()
        sp_layout = QVBoxLayout(sp_container)
        sp_layout.addWidget(self.sp_toolbar)
        sp_layout.addWidget(self.sp_canvas)
        self.graph_stack.addWidget(sp_container)

        # ===== GABUNGKAN KOMPONEN =====
        graphics_layout.addWidget(self.btn_container)
        graphics_layout.addWidget(self.graph_stack)
        graphics_layout.setStretch(0, 1)
        graphics_layout.setStretch(1, 10)
        
        right_layout.addWidget(self.graphics_box)
        self.right_container.show()
        
        # Tampilkan default graph
        self.show_default_graph()

        # Training Box Settings
        self.training_box_settings = QGroupBox("Model Configuration")
        box_layout = QVBoxLayout(self.training_box_settings)
        
        scroll_widget = QWidget()
        settings_layout = QVBoxLayout(scroll_widget)
        
        # Preprocessing Settings
        preprocessing_label = QLabel("Preprocessing Settings")
        preprocessing_label.setStyleSheet("font-weight: bold")
        self.checkbox_minmax = QCheckBox("MinMaxScaler")
        self.checkbox_standard = QCheckBox("StandardScaler")
        # self.column_selection = QComboBox()
        # self.column_selection.addItems(["Pilih kolom", "Fitur", "Target"])

        settings_layout.addWidget(preprocessing_label)
        settings_layout.addWidget(self.checkbox_minmax)
        settings_layout.addWidget(self.checkbox_standard)
        # settings_layout.addWidget(QLabel("Jenis Kolom:"))
        # settings_layout.addWidget(self.column_selection)

        # Model Selection
        model_label = QLabel("Pilih Jenis Model")
        model_label.setStyleSheet("font-weight: bold")
        self.checkbox_rf_classifier = QCheckBox("Random Forest Classifier")
        self.checkbox_rf_regressor = QCheckBox("Random Forest Regressor")

        settings_layout.addWidget(model_label)
        settings_layout.addWidget(self.checkbox_rf_classifier)
        settings_layout.addWidget(self.checkbox_rf_regressor)

        # Parameter Settings
        rf_label = QLabel("Parameter Setting Random Forest")
        rf_label.setStyleSheet("font-weight: bold")

        self.n_estimators_input = QLineEdit()
        self.n_estimators_input.setPlaceholderText("n_estimators (misal 100)")
        self.max_depth_input = QLineEdit()
        self.max_depth_input.setPlaceholderText("max_depth (misal 10)")
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["squared_error", "absolute_error"])
        self.regressor_combo = QComboBox()
        self.regressor_combo.addItems(["Remaining Useful Life", "Durasi Downtime"])

        settings_layout.addWidget(rf_label)
        settings_layout.addWidget(QLabel("n_estimators:"))
        settings_layout.addWidget(self.n_estimators_input)
        settings_layout.addWidget(QLabel("max_depth:"))
        settings_layout.addWidget(self.max_depth_input)
        settings_layout.addWidget(QLabel("Criterion:"))
        settings_layout.addWidget(self.criterion_combo)
        settings_layout.addWidget(QLabel("Error Metric:"))
        settings_layout.addWidget(self.metric_combo)
        settings_layout.addWidget(QLabel("Pilih Target (untuk Random Forest Regressor):"))
        settings_layout.addWidget(self.regressor_combo)

        # Buttons
        button_layout = QHBoxLayout()
        self.label_button = QPushButton("Labeling")
        self.label_button.setStyleSheet("background-color: #F79B72; color: white; padding: 4px;")
        self.label_button.clicked.connect(self.show_label_data)
        self.split_button = QPushButton("Split Data")
        self.split_button.setStyleSheet("background-color: #2196F3; color: white; padding: 4px;")
        self.split_button.clicked.connect(self.show_split_dialog)  # New connection
        self.train_button = QPushButton("Valid & Train")
        self.train_button.setStyleSheet("background-color: #4CAF50; color: black; padding: 4px;")
        self.train_button.clicked.connect(self.show_training_option_dialog)
        self.load_button = QPushButton("Load Model")
        self.load_button.setStyleSheet("background-color: #FFC107; color: black; padding: 4px;")
        
        button_layout.addWidget(self.label_button)
        button_layout.addWidget(self.split_button)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.load_button)
        settings_layout.addLayout(button_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        box_layout.addWidget(scroll_area)

        # Add widgets to containers
        container_layout.addWidget(self.training_box_model)
        container_layout.addWidget(self.training_box_model_regressor)
        container_layout.addWidget(self.training_box_db)
        settingModel_layout.addWidget(self.training_box_settings)

        # Show all widgets
        self.training_widgets = [
            self.training_label,
            self.training_container,
            self.right_container,
            self.model_container,
            self.terminal_monitor
        ]

        for widget in self.training_widgets:
            widget.show()

        self.connect_to_database()
    
    def show_default_graph(self):
        """Menampilkan contoh grafik saat pertama kali dijalankan"""
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)
        
        # Contoh confusion matrix default
        cm = np.array([[50, 10], [5, 35]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Contoh Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        self.cm_canvas.draw()
        
        # Contoh scatter plot default
        self.sp_figure.clear()
        ax = self.sp_figure.add_subplot(111)
        x = np.random.rand(50) * 100
        y = x + np.random.normal(0, 10, 50)
        ax.scatter(x, y)
        ax.plot([0, 100], [0, 100], 'r--')
        ax.set_title('Contoh Scatter Plot')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
        self.sp_canvas.draw()
        
        # Set default view
        self.graph_stack.setCurrentIndex(0)
        self.update_button_styles(0)

    def show_classifier_graph(self):
        """Menampilkan confusion matrix aktual"""
        if not hasattr(self, 'classifier_model'):
            QMessageBox.warning(self, "Peringatan", "Model classifier belum dilatih!")
            return
        
        try:
            self.graph_stack.setCurrentIndex(0)
            self.update_button_styles(0)
            
            y_pred = self.classifier_model.predict(self.X_test)
            self.plot_confusion_matrix(self.y_test, y_pred)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menampilkan grafik: {str(e)}")

    def show_regressor_graph(self):
        """Menampilkan scatter plot aktual"""
        if not hasattr(self, 'regressor_model'):
            QMessageBox.warning(self, "Peringatan", "Model regressor belum dilatih!")
            return
        
        try:
            self.graph_stack.setCurrentIndex(1)
            self.update_button_styles(1)
            
            y_pred = self.regressor_model.predict(self.X_test)
            self.plot_scatter(self.y_test, y_pred)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal menampilkan grafik: {str(e)}")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix aktual"""
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        self.cm_canvas.draw()

    def plot_scatter(self, y_true, y_pred):
        """Plot scatter plot aktual"""
        self.sp_figure.clear()
        ax = self.sp_figure.add_subplot(111)
        
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 'r--')
        ax.set_title('Actual vs Predicted')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
        self.sp_canvas.draw()

    def update_button_styles(self, active_index):
        """Update style tombol berdasarkan yang aktif"""
        active_style = """
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            font-size: 12px;
        """
        inactive_style = """
            background-color: #757575;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            font-size: 12px;
        """
        
        self.classifier_btn.setStyleSheet(active_style if active_index == 0 else inactive_style)
        self.regressor_btn.setStyleSheet(active_style if active_index == 1 else inactive_style)

    def show_split_dialog(self):
        """Menampilkan dialog split data dengan opsi kolom target"""
        try:
            # Dapatkan koneksi database dari fungsi connect_to_db
            conn = self.connect_to_db()
            if conn is None:
                QMessageBox.critical(self, "Error", "Tidak dapat terhubung ke database!")
                return

            # Ambil jumlah data dari database
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN kondisi_motor IS NOT NULL THEN 1 ELSE 0 END) as count_class,
                    SUM(CASE WHEN downtime_motor IS NOT NULL THEN 1 ELSE 0 END) as count_reg
                FROM tubes
            """)
            counts = cursor.fetchone()
            cursor.close()
            conn.close()  # Tutup koneksi setelah selesai
            
            if counts[0] == 0 and counts[1] == 0:
                QMessageBox.warning(self, "Peringatan", "Tidak ada data yang bisa dibagi!")
                return
                
            # Buat dialog custom
            dialog = QDialog(self)
            dialog.setWindowTitle("Pembagian Data")
            dialog.setFixedSize(450, 300)
            
            layout = QVBoxLayout()
            
            # Pilihan Kolom Target
            target_group = QGroupBox("Pilih Kolom Target")
            target_layout = QVBoxLayout()
            
            self.target_combobox = QComboBox()
            if counts[0] > 0:
                self.target_combobox.addItem("kondisi_motor (Klasifikasi)", "class")
            if counts[1] > 0:
                self.target_combobox.addItem("downtime_motor (Regresi)", "reg")
                
            target_layout.addWidget(self.target_combobox)
            target_group.setLayout(target_layout)
            layout.addWidget(target_group)
            
            # Info jumlah data
            info_label = QLabel()
            if counts[0] > 0 and counts[1] > 0:
                info_label.setText(
                    f"Data tersedia:\n"
                    f"- Kondisi Motor: {counts[0]} records\n"
                    f"- Downtime Motor: {counts[1]} records"
                )
            else:
                info_label.setText(f"Data tersedia: {max(counts)} records")
            layout.addWidget(info_label)
            
            # Slider
            slider_group = QGroupBox("Pengaturan Pembagian")
            slider_layout = QVBoxLayout()
            
            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel("Test Size:"))
            
            self.test_size_slider = QSlider(Qt.Orientation.Horizontal)
            self.test_size_slider.setRange(10, 40)
            self.test_size_slider.setValue(20)
            h_layout.addWidget(self.test_size_slider)
            
            self.test_size_label = QLabel("20%")
            h_layout.addWidget(self.test_size_label)
            slider_layout.addLayout(h_layout)
            
            # Checkbox untuk shuffle dan stratify
            self.shuffle_check = QCheckBox("Acak Data (Shuffle)")
            self.shuffle_check.setChecked(True)
            slider_layout.addWidget(self.shuffle_check)
            
            self.stratify_check = QCheckBox("Pertahankan Distribusi (Stratify)")
            self.stratify_check.setChecked(True)
            slider_layout.addWidget(self.stratify_check)
            
            slider_group.setLayout(slider_layout)
            layout.addWidget(slider_group)
            
            # Tombol
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            dialog.setLayout(layout)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                test_size = self.test_size_slider.value() / 100
                target_type = self.target_combobox.currentData()
                shuffle = self.shuffle_check.isChecked()
                stratify = self.stratify_check.isChecked() if target_type == "class" else False
                
                self.execute_data_split(test_size, target_type, shuffle, stratify)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal memproses data:\n{str(e)}")

    def save_models_to_file(self):
        """Menyimpan model classifier, regressor, dan scaler ke file dengan timestamp, 
        serta menghapus file lama jika ada"""
        from datetime import datetime
        from pathlib import Path

        try:
            # Buat folder simpan model jika belum ada
            os.makedirs("saved_models", exist_ok=True)

            # Hapus file lama jika ada
            if hasattr(self, 'classifier_model_path') and isinstance(self.classifier_model_path, str):
                old_classifier = Path(self.classifier_model_path)
                if old_classifier.exists():
                    old_classifier.unlink()
                    self.terminal_output.append(f"File lama classifier dihapus: {old_classifier}")

            if hasattr(self, 'regressor_model_path') and isinstance(self.regressor_model_path, str):
                old_regressor = Path(self.regressor_model_path)
                if old_regressor.exists():
                    old_regressor.unlink()
                    self.terminal_output.append(f"File lama regressor dihapus: {old_regressor}")

            # Timestamp baru
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Path baru
            classifier_path = f"saved_models/classifier_{timestamp}.joblib"
            regressor_path = f"saved_models/regressor_{timestamp}.joblib"
            scaler_path = f"saved_models/scaler_{timestamp}.joblib"

            # Simpan classifier
            if hasattr(self, 'classifier_model') and self.classifier_model is not None:
                joblib.dump(self.classifier_model, classifier_path)
                self.classifier_model_path = classifier_path
                self.terminal_output.append(f"Classifier model disimpan di: {classifier_path}")
                if hasattr(self, 'classifier_label'):
                    self.classifier_label.setText(f"File hasil training model classifier:\n{classifier_path}")

            # Simpan regressor
            if hasattr(self, 'regressor_model') and self.regressor_model is not None:
                joblib.dump(self.regressor_model, regressor_path)
                self.regressor_model_path = regressor_path
                self.terminal_output.append(f"Regressor model disimpan di: {regressor_path}")
                if hasattr(self, 'classifier_label'):
                    self.regressor_label.setText(f"File hasil training model regressor:\n{regressor_path}")
            # Simpan scaler
            if hasattr(self, 'scaler') and self.scaler is not None:
                joblib.dump(self.scaler, scaler_path)
                self.terminal_output.append(f"Scaler disimpan di: {scaler_path}")

            return True

        except Exception as e:
            self.terminal_output.append(f"Gagal menyimpan model: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocessing data sebelum training"""
        # Simpan nama fitur jika ada
        self.feature_names = list(self.X_train.columns) if hasattr(self.X_train, 'columns') else None
        
        # Konversi ke numpy array
        X_train = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
        X_test = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
        
        return X_train, X_test

    def train_model(self):
        
        """Melatih model dengan penanganan error lengkap dan update UI"""
        try:
            # ===== 1. SETUP AWAL =====
            self.set_notification_icon("notification off.png")
            self.terminal_output.append("Memulai proses training...")
            
            # ===== VALIDASI DATA LENGKAP =====
            # Validasi data training eksis
            if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
                raise ValueError("Data training belum dimuat!")
            
            # Validasi data test eksis jika diperlukan
            if not hasattr(self, 'X_test'):
                raise ValueError("Data test belum dimuat!")
            
            # Validasi dimensi data
            if len(self.X_train) != len(self.y_train):
                raise ValueError("Jumlah sampel X_train dan y_train tidak sama!")
            
            # Validasi setidaknya ada 1 sampel data
            if len(self.X_train) == 0:
                raise ValueError("Data training kosong!")
            
            # Validasi setidaknya ada 1 fitur
            if self.X_train.shape[1] == 0:
                raise ValueError("Tidak ada fitur yang tersedia untuk training!")
            
            # Validasi target tidak kosong
            if len(np.unique(self.y_train)) == 0:
                raise ValueError("Target training kosong!")
            
            # Validasi untuk klasifikasi: minimal 2 kelas
            if self.checkbox_rf_classifier.isChecked():
                unique_classes = np.unique(self.y_train)
                if len(unique_classes) < 2:
                    raise ValueError("Untuk klasifikasi, dibutuhkan minimal 2 kelas dalam data training!")
            
            # Validasi parameter model
            try:
                n_estimators = int(self.n_estimators_input.text()) if self.n_estimators_input.text() else 100
                if n_estimators <= 0:
                    raise ValueError("n_estimators harus bilangan positif!")
                
                max_depth = int(self.max_depth_input.text()) if self.max_depth_input.text() else None
                if max_depth is not None and max_depth <= 0:
                    raise ValueError("max_depth harus None atau bilangan positif!")
            except ValueError as e:
                raise ValueError(f"Parameter tidak valid: {str(e)}")
            
            # Validasi setidaknya satu model dipilih
            if not self.checkbox_rf_classifier.isChecked() and not self.checkbox_rf_regressor.isChecked():
                raise ValueError("Pilih minimal satu model (Classifier atau Regressor)!")
            
            # Simpan nama fitur sebelum konversi
            self.feature_names = list(self.X_train.columns) if hasattr(self.X_train, 'columns') else None

            # ===== 2. PREPROCESSING =====
            preprocessing = 'minmax' if self.checkbox_minmax.isChecked() else 'standard' if self.checkbox_standard.isChecked() else None
            
            # Validasi kolom kategorikal
            categorical_cols = self.X_train.select_dtypes(include=['object']).columns
            for column in categorical_cols:
                if len(self.X_train[column].unique()) > 100:
                    raise ValueError(f"Kolom {column} memiliki terlalu banyak nilai unik (>100) untuk encoding!")
            
            # Label encoding untuk kolom kategorikal
            label_encoders = {}
            for column in categorical_cols:
                le = LabelEncoder()
                self.X_train[column] = le.fit_transform(self.X_train[column].astype(str))
                self.X_test[column] = le.transform(self.X_test[column].astype(str))
                label_encoders[column] = le

            # Konversi ke numpy array untuk konsistensi
            X_train = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
            X_test = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test

            # Validasi tidak ada NaN setelah preprocessing
            if np.isnan(X_train).any() or np.isnan(self.y_train).any():
                raise ValueError("Data mengandung nilai NaN setelah preprocessing!")
            
            if np.isnan(X_test).any():
                raise ValueError("Data test mengandung nilai NaN setelah preprocessing!")

            # Scaling data
            if preprocessing == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                self.scaler = MinMaxScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            elif preprocessing == 'standard':
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                self.scaler = None
            
            # ===== 3. PERSIAPAN TRAINING =====
            n_estimators = int(self.n_estimators_input.text()) if self.n_estimators_input.text() else 100
            max_depth = int(self.max_depth_input.text()) if self.max_depth_input.text() else None
            
            # Reset UI
            self.reset_ui_elements()

            # ===== 4. TRAINING CLASSIFIER =====
            if self.checkbox_rf_classifier.isChecked():
                from sklearn.ensemble import RandomForestClassifier
                criterion = self.criterion_combo.currentText()
                
                self.classifier_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    criterion=criterion,
                    random_state=42
                )
                
                try:
                    self.terminal_output.append("Memulai training classifier...")
                    self.classifier_model.fit(X_train_scaled, self.y_train)
                    # Simpan data format yang digunakan
                    self.classifier_data_format = 'numpy'
                    
                    # Update UI
                    self.update_training_results()
                    self.terminal_output.append("Training classifier berhasil!")
                    
                except Exception as e:
                    raise RuntimeError(f"Gagal training classifier: {str(e)}")

            # ===== 5. TRAINING REGRESSOR =====
            if self.checkbox_rf_regressor.isChecked():
                from sklearn.ensemble import RandomForestRegressor
                criterion = 'squared_error' if self.metric_combo.currentText() == 'mse' else 'absolute_error'
                
                self.regressor_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    criterion=criterion,
                    random_state=42
                )
                
                try:
                    self.terminal_output.append("Memulai training regressor...")
                    self.regressor_model.fit(X_train_scaled, self.y_train)
                    # Simpan data format yang digunakan
                    self.regressor_data_format = 'numpy'
                    
                    # Update UI
                    self.update_training_results()
                    self.terminal_output.append("Training regressor berhasil!")
                    
                except Exception as e:
                    raise RuntimeError(f"Gagal training regressor: {str(e)}")

            # ===== 6. VALIDASI MODEL SETELAH TRAINING =====
            if self.checkbox_rf_classifier.isChecked() and hasattr(self, 'classifier_model'):
                # Validasi classifier
                from sklearn.metrics import accuracy_score, classification_report
                
                # Prediksi pada data training
                train_pred = self.classifier_model.predict(X_train_scaled)
                train_acc = accuracy_score(self.y_train, train_pred)
                
                # Prediksi pada data test (jika ada y_test)
                if hasattr(self, 'y_test') and self.y_test is not None:
                    test_pred = self.classifier_model.predict(X_test_scaled)
                    test_acc = accuracy_score(self.y_test, test_pred)
                    self.terminal_output.append(f"Akurasi Training: {train_acc:.4f}, Akurasi Test: {test_acc:.4f}")
                    
                    # Validasi overfitting
                    if train_acc > 0.95 and (train_acc - test_acc) > 0.2:
                        self.terminal_output.append("Peringatan: Model mungkin overfitting!")
                else:
                    self.terminal_output.append(f"Akurasi Training: {train_acc:.4f}")
                
                # Laporan klasifikasi
                self.terminal_output.append("Laporan Klasifikasi:")
                self.terminal_output.append(str(classification_report(self.y_train, train_pred)))
                
                # Validasi feature importance
                if hasattr(self.classifier_model, 'feature_importances_'):
                    self.terminal_output.append("\nFeature Importance:")
                    for name, importance in zip(self.feature_names, self.classifier_model.feature_importances_):
                        self.terminal_output.append(f"{name}: {importance:.4f}")
                
            if self.checkbox_rf_regressor.isChecked() and hasattr(self, 'regressor_model'):
                # Validasi regressor
                from sklearn.metrics import mean_squared_error, r2_score
                
                # Prediksi pada data training
                train_pred = self.regressor_model.predict(X_train_scaled)
                train_mse = mean_squared_error(self.y_train, train_pred)
                train_r2 = r2_score(self.y_train, train_pred)
                
                # Prediksi pada data test (jika ada y_test)
                if hasattr(self, 'y_test') and self.y_test is not None:
                    test_pred = self.regressor_model.predict(X_test_scaled)
                    test_mse = mean_squared_error(self.y_test, test_pred)
                    test_r2 = r2_score(self.y_test, test_pred)
                    
                    self.terminal_output.append(f"Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
                    self.terminal_output.append(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
                    
                    # Validasi overfitting
                    if (train_r2 > 0.9) and (train_r2 - test_r2) > 0.3:
                        self.terminal_output.append("Peringatan: Model mungkin overfitting!")
                else:
                    self.terminal_output.append(f"Training MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
                
                # Validasi feature importance
                if hasattr(self.regressor_model, 'feature_importances_'):
                    self.terminal_output.append("\nFeature Importance:")
                    for name, importance in zip(self.feature_names, self.regressor_model.feature_importances_):
                        self.terminal_output.append(f"{name}: {importance:.4f}")
            
            # ===== 7. SIMPAN MODEL DAN HASIL =====
            self.save_models_to_file()
            
            # Simpan metrik evaluasi
            self.model_metrics = {
                'classifier': {
                    'train_accuracy': train_acc if self.checkbox_rf_classifier.isChecked() else None,
                    'test_accuracy': test_acc if (self.checkbox_rf_classifier.isChecked() and hasattr(self, 'y_test')) else None
                },
                'regressor': {
                    'train_mse': train_mse if self.checkbox_rf_regressor.isChecked() else None,
                    'train_r2': train_r2 if self.checkbox_rf_regressor.isChecked() else None,
                    'test_mse': test_mse if (self.checkbox_rf_regressor.isChecked() and hasattr(self, 'y_test')) else None,
                    'test_r2': test_r2 if (self.checkbox_rf_regressor.isChecked() and hasattr(self, 'y_test')) else None
                }
            }
            
            # ===== 8. UPDATE UI AKHIR =====
            self.set_notification_icon("notification on.png")
            QMessageBox.information(self, "Sukses", "Training dan validasi model berhasil!")
            self.terminal_output.append("Training dan validasi berhasil")
            
            # Update plot jika diperlukan
            if hasattr(self, 'update_plots'):
                self.update_plots()

        except ValueError as ve:
            QMessageBox.critical(self, "Input Error", f"Parameter tidak valid:\n{str(ve)}")
            self.terminal_output.append(f"Error: {str(ve)}")
        except RuntimeError as re:
            QMessageBox.critical(self, "Training Error", f"Proses training gagal:\n{str(re)}")
            self.terminal_output.append(f"Error: {str(re)}")
        except Exception as e:
            QMessageBox.critical(self, "System Error", f"Error tidak terduga:\n{str(e)}")
            self.terminal_output.append(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def predict(self, model_type='classifier'):
        """Prediksi dengan format data yang konsisten"""
        model = self.classifier_model if model_type == 'classifier' else self.regressor_model
        
        if not hasattr(self, 'X_test'):
            raise ValueError("Data test belum dimuat")
            
        # Konversi ke numpy array untuk konsistensi
        X_test = self.X_test.values if hasattr(self.X_test, 'values') else self.X_test
        
        # Jika ada scaler, apply scaling
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
            
        return model.predict(X_test)

    def update_training_results(self):
        """Update semua label hasil training di UI"""
        # ===== CLASSIFIER =====
        if hasattr(self, 'classifier_model') and self.classifier_model is not None:
            try:
                # Clear layout
                self.clear_layout(self.training_box_model.layout())
                layout = self.training_box_model.layout()
                
                # Prediksi dan hitung metrik
                y_pred = self.predict('classifier')
                report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
                
                # Update UI
                layout.addWidget(QLabel("Model: Random Forest Classifier"))
                layout.addWidget(QLabel(f"Akurasi: {report['accuracy']:.2%}"))
                layout.addWidget(QLabel(f"Presisi: {report['weighted avg']['precision']:.2%}"))
                layout.addWidget(QLabel(f"Recall: {report['weighted avg']['recall']:.2%}"))
                layout.addWidget(QLabel(f"F1-Score: {report['weighted avg']['f1-score']:.2%}"))
                
                # Feature importance
                if hasattr(self.classifier_model, 'feature_importances_') and self.feature_names:
                    importances = self.classifier_model.feature_importances_
                    top_idx = importances.argmax()
                    layout.addWidget(QLabel(f"Top Fitur: {self.feature_names[top_idx]}"))
                
                # Visualisasi
                self.plot_confusion_matrix(self.y_test, y_pred)
                
            except Exception as e:
                print(f"Error update classifier UI: {str(e)}")

        # ===== REGRESSOR =====
        if hasattr(self, 'regressor_model') and self.regressor_model is not None:
            try:
                # Clear layout
                self.clear_layout(self.training_box_model_regressor.layout())
                layout = self.training_box_model_regressor.layout()
                
                # Prediksi dan hitung metrik
                y_pred = self.predict('regressor')
                
                # Update UI
                layout.addWidget(QLabel("Model: Random Forest Regressor"))
                layout.addWidget(QLabel(f"MSE: {mean_squared_error(self.y_test, y_pred):.4f}"))
                layout.addWidget(QLabel(f"MAE: {mean_absolute_error(self.y_test, y_pred):.4f}"))
                layout.addWidget(QLabel(f"RMSE: {mean_squared_error(self.y_test, y_pred)**0.5:.4f}"))
                layout.addWidget(QLabel(f"R² Score: {r2_score(self.y_test, y_pred):.4f}"))
                
                # Feature importance
                if hasattr(self.regressor_model, 'feature_importances_') and self.feature_names:
                    importances = self.regressor_model.feature_importances_
                    top_idx = importances.argmax()
                    layout.addWidget(QLabel(f"Top Fitur: {self.feature_names[top_idx]}"))
                
                # Visualisasi
                self.plot_scatter(self.y_test, y_pred)
                
            except Exception as e:
                print(f"Error update regressor UI: {str(e)}")

    def clear_layout(self, layout):
        """Hapus semua widget dari layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

    def plot_feature_importance(self, importances, feature_names):
        """Plot feature importance (jika ada matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            
            # Clear previous plot
            if hasattr(self, 'feature_importance_canvas'):
                self.feature_importance_layout.removeWidget(self.feature_importance_canvas)
                self.feature_importance_canvas.deleteLater()
            
            # Create new plot
            fig, ax = plt.subplots(figsize=(8, 4))
            indices = range(len(importances))
            names = feature_names if feature_names is not None else [f"Feat {i}" for i in indices]
            
            ax.barh(indices, importances, align='center')
            ax.set_yticks(indices)
            ax.set_yticklabels(names)
            ax.set_title("Feature Importance")
            
            # Embed plot in UI
            self.feature_importance_canvas = FigureCanvasQTAgg(fig)
            self.feature_importance_layout.addWidget(self.feature_importance_canvas)
            self.feature_importance_canvas.draw()
            
        except Exception as e:
            print(f"Gagal membuat plot: {str(e)}")

    def reset_ui_elements(self):
        """Reset semua UI elements sebelum training"""
        # Reset Classifier UI
        self.clear_layout(self.training_box_model.layout())
        model_layout = self.training_box_model.layout()
        model_layout.addWidget(QLabel("Model: Random Forest Classifier"))
        model_layout.addWidget(QLabel("Akurasi: -%"))
        model_layout.addWidget(QLabel("Presisi: -%"))
        model_layout.addWidget(QLabel("Recall: -%"))
        model_layout.addWidget(QLabel("F1-Score: -%"))
        model_layout.addWidget(QLabel("Top Fitur: -"))

        # Reset Regressor UI
        self.clear_layout(self.training_box_model_regressor.layout())
        model2_layout = self.training_box_model_regressor.layout()
        model2_layout.addWidget(QLabel("Model: Random Forest Regressor"))
        model2_layout.addWidget(QLabel("MSE: -"))
        model2_layout.addWidget(QLabel("MAE: -"))
        model2_layout.addWidget(QLabel("RMSE: -"))
        model2_layout.addWidget(QLabel("R² Score: -"))
        model2_layout.addWidget(QLabel("Top Fitur: -"))

        # Reset tambahan (jika ada)
        if hasattr(self, 'feature_importance_canvas'):
            self.feature_importance_layout.removeWidget(self.feature_importance_canvas)
            self.feature_importance_canvas.deleteLater()

    def clear_layout(self, layout):
        """Helper untuk menghapus semua widget dari layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

    def show_split_dialog(self):
        """Menampilkan dialog split data dengan opsi kolom target"""
        try:
            # Dapatkan koneksi database dari fungsi connect_to_db
            conn = self.connect_to_db()
            if conn is None:
                QMessageBox.critical(self, "Error", "Tidak dapat terhubung ke database!")
                return

            # Ambil jumlah data dari database
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN kondisi_motor IS NOT NULL THEN 1 ELSE 0 END) as count_class,
                    SUM(CASE WHEN downtime_motor IS NOT NULL THEN 1 ELSE 0 END) as count_reg
                FROM tubes
            """)
            counts = cursor.fetchone()
            cursor.close()
            conn.close()  # Tutup koneksi setelah selesai
            
            if counts is None or (counts[0] == 0 and counts[1] == 0):
                QMessageBox.warning(self, "Peringatan", "Tidak ada data yang bisa dibagi!")
                return
                
            # Buat dialog custom
            dialog = QDialog(self)
            dialog.setWindowTitle("Pembagian Data")
            dialog.setFixedSize(450, 300)
            
            layout = QVBoxLayout()
            
            # Pilihan Kolom Target
            target_group = QGroupBox("Pilih Kolom Target")
            target_layout = QVBoxLayout()
            
            self.target_combobox = QComboBox()
            if counts[0] > 0:
                self.target_combobox.addItem("kondisi_motor (Klasifikasi)", "class")
            if counts[1] > 0:
                self.target_combobox.addItem("downtime_motor (Regresi)", "reg")
                
            target_layout.addWidget(self.target_combobox)
            target_group.setLayout(target_layout)
            layout.addWidget(target_group)
            
            # Info jumlah data
            info_label = QLabel()
            if counts[0] > 0 and counts[1] > 0:
                info_label.setText(
                    f"Data tersedia:\n"
                    f"- Kondisi Motor: {counts[0]} records\n"
                    f"- Downtime Motor: {counts[1]} records"
                )
            else:
                available_count = counts[0] if counts[0] > 0 else counts[1]
                info_label.setText(f"Data tersedia: {available_count} records")
            layout.addWidget(info_label)
            
            # Slider
            slider_group = QGroupBox("Pengaturan Pembagian")
            slider_layout = QVBoxLayout()

            h_layout = QHBoxLayout()
            h_layout.addWidget(QLabel("Test Size:"))

            self.test_size_slider = QSlider(Qt.Orientation.Horizontal)
            self.test_size_slider.setRange(10, 40)
            self.test_size_slider.setValue(20)
            self.test_size_slider.valueChanged.connect(self.update_test_size_label)  # ✅ Tambahkan ini
            h_layout.addWidget(self.test_size_slider)

            self.test_size_label = QLabel("20%")
            h_layout.addWidget(self.test_size_label)
            slider_layout.addLayout(h_layout)

            # Checkbox untuk shuffle dan stratify
            self.shuffle_check = QCheckBox("Acak Data (Shuffle)")
            self.shuffle_check.setChecked(True)
            slider_layout.addWidget(self.shuffle_check)

            self.stratify_check = QCheckBox("Pertahankan Distribusi (Stratify)")
            self.stratify_check.setChecked(True)
            self.stratify_check.setEnabled(counts[0] > 0)  # Hanya aktif untuk klasifikasi
            slider_layout.addWidget(self.stratify_check)

            slider_group.setLayout(slider_layout)
            layout.addWidget(slider_group)

            # Tombol
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            dialog.setLayout(layout)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                test_size = self.test_size_slider.value() / 100
                target_type = self.target_combobox.currentData()
                shuffle = self.shuffle_check.isChecked()
                stratify = self.stratify_check.isChecked() if target_type == "class" else False

                self.execute_data_split(test_size, target_type, shuffle, stratify)

                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Gagal memproses data:\n{str(e)}\n\nPastikan koneksi database valid."
            )

    def update_test_size_label(self, value):
        self.test_size_label.setText(f"{value}%")

    def execute_data_split(self, test_size, target_type, shuffle=True, stratify=True):
        """Eksekusi pembagian data dengan manajemen koneksi yang aman"""
        try:
            conn = self.connect_to_db()
            if conn is None:
                QMessageBox.critical(self, "Error", "Tidak dapat terhubung ke database!")
                return

            # Tentukan query berdasarkan target type
            if target_type == "class":
                query = """
                SELECT temperature, vibration, kondisi_motor 
                FROM tubes 
                WHERE kondisi_motor IS NOT NULL
                """
                target_col = 'kondisi_motor'
            else:
                query = """
                SELECT temperature, vibration, downtime_motor 
                FROM tubes 
                WHERE downtime_motor IS NOT NULL
                """
                target_col = 'downtime_motor'
            
            # Eksekusi query
            cursor = conn.cursor()
            df = pd.read_sql(query, conn)
            cursor.close()
            conn.close()
            
            if df.empty:
                QMessageBox.warning(self, "Peringatan", "Tidak ada data yang sesuai!")
                return
                
            X = df[['temperature', 'vibration',]]
            y = df[target_col]
            
            # Split data
            if stratify and target_type == "class":
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    shuffle=shuffle,
                    stratify=y
                )
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    shuffle=shuffle
                )
            
            # Update UI
            self.terminal_output.append(
                f"\n[SPLIT DATA] Berhasil membagi data:\n"
                f"- Target: {target_col}\n"
                f"- Training: {len(self.X_train)} records\n"
                f"- Testing: {len(self.X_test)} records\n"
                f"- Rasio: {int(test_size*100)}% testing\n"
                f"- Shuffle: {'Ya' if shuffle else 'Tidak'}\n"
                f"- Stratify: {'Ya' if stratify and target_type == 'class' else 'Tidak'}"
            )
            
            # Update model info box
            if target_type == "class":
                self.training_box_model.setTitle("Hasil Training (Klasifikasi)")
            else:
                self.training_box_model_regressor.setTitle("Hasil Training (Regresi)")
            
            QMessageBox.information(
                self,
                "Berhasil",
                f"Data berhasil dibagi berdasarkan {target_col}:\n\n"
                f"Training: {len(self.X_train)} records\n"
                f"Testing: {len(self.X_test)} records",
                QMessageBox.StandardButton.Ok
            )
            
        except Exception as e:
            self.terminal_output.append(f"\n[ERROR] Gagal split data: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Gagal membagi data:\n{str(e)}",
                QMessageBox.StandardButton.Ok
            )

    def show_label_data(self):
        # Konfirmasi sebelum mulai labeling
        reply = QMessageBox.question(
            self,
            "Konfirmasi Labeling",
            "Apakah Anda yakin ingin mulai proses labeling?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            self.terminal_output.append("Proses labeling dibatalkan oleh pengguna.")
            return

        try:
            # Gunakan koneksi dari self.connect_to_db()
            conn = self.connect_to_db()
            cursor = conn.cursor()

            # Cek apakah kolom label_kode sudah ada, jika belum tambahkan
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('tubes') AND name = 'label_kode')
            BEGIN
                ALTER TABLE tubes ADD label_kode VARCHAR(10) NULL
            END
            """)
            conn.commit()

            # Ambil data yang kondisi_motor atau downtime_motor masih NULL
            query = """
            SELECT * FROM tubes
            WHERE kondisi_motor IS NULL OR downtime_motor IS NULL OR label_kode IS NULL
            """
            df = pd.read_sql(query, conn)

            if df.empty:
                self.terminal_output.append("Tidak ada data yang perlu dilabeli.")
                return

            def determine_condition(temp, vib):
                """
                Rule-based labeling dengan prioritas kondisi khusus dan expandable rules
                Returns:
                    (status, priority, label_code)
                """
                # ========== DEFINE THRESHOLDS ==========
                TEMP_CRITICAL_HIGH = 60
                TEMP_HIGH = 45
                TEMP_NORMAL_MAX = 30
                TEMP_LOW = 25
                
                VIB_DANGER = 15
                VIB_HIGH = 11
                VIB_NORMAL_MAX = 10
                VIB_LOW = 5

                # ===== KONDISI KHUSUS PRIORITAS TERTINGGI =====
                # 1. Kombinasi ekstrim
                if temp > TEMP_CRITICAL_HIGH and vib > VIB_DANGER:
                    return 'suhu kritis, getaran bahaya', 1, 'C1'
                
                if temp > TEMP_HIGH and vib < VIB_NORMAL_MAX:
                    return 'suhu tinggi, getaran rendah', 2, 'C2'
                
                if temp < TEMP_LOW and vib > VIB_HIGH:
                    return 'suhu rendah, getaran tinggi', 2, 'C2'
                
                if vib > VIB_DANGER:
                    return 'getaran bahaya', 3, 'C3'
                
                if temp < TEMP_LOW:
                    return 'suhu rendah', 4, 'C4'
                
                # ===== RULE UTAMA =====
                if temp < TEMP_NORMAL_MAX or vib < VIB_NORMAL_MAX:
                    return 'normal', 5, 'C5'
                
                if (TEMP_NORMAL_MAX <= temp <= TEMP_HIGH) or (VIB_NORMAL_MAX <= vib <= VIB_HIGH):
                    return 'perlu perawatan', 4, 'C4'
                
                # ===== DEFAULT =====
                return 'indikasi rusak', 3, 'C3'
                
            # Apply labeling and unpack all three values
            labels = df.apply(lambda row: determine_condition(row['temperature'], row['vibration']), axis=1)
            df[['kondisi_motor', 'downtime_motor', 'label_kode']] = pd.DataFrame(labels.tolist(), index=df.index)

            # Update hasil ke database
            for _, row in df.iterrows():
                update_query = """
                UPDATE tubes
                SET kondisi_motor = ?, downtime_motor = ?, label_kode = ?
                WHERE timestamp = ?
                """
                cursor.execute(update_query, 
                            (row['kondisi_motor'], 
                            row['downtime_motor'],
                            row['label_kode'],
                            row['timestamp']))
                self.terminal_output.append(f"Labeling data {row['timestamp']}: {row['label_kode']} - {row['kondisi_motor']}")

            conn.commit()
            self.terminal_output.append("Labeling selesai dan data diperbarui di database.")
            QMessageBox.information(self, "Sukses", "Labeling berhasil dan data telah diperbarui.")

        except Exception as e:
            self.terminal_output.append(f"Terjadi kesalahan saat labeling: {e}")
            QMessageBox.critical(self, "Error", f"Gagal melakukan labeling:\n{e}")
        finally:
            if 'conn' in locals():
                conn.close()

    
    def show_vibration(self):
        self.clear_content()
        self.serial_port = None  # pyserial Serial instance
        
        # Create the label
        self.section_title = QLabel("VIBRATION SECTION", self)
        # Manually position the label at x=50, y=30
        self.section_title.move(20, 70)
        # Optionally you can set fixed size or font to adjust appearance
        self.section_title.resize(200, 40)  # width, height
        font = self.section_title.font()
        font.setPointSize(10)
        font.setBold(True)
        self.section_title.setFont(font)
        self.section_title.show()

        # Below charts: Serial connection controls container
        self.serial_control_container = QWidget(self)
        self.serial_control_container.setGeometry(20, 90, 860, 60)
        self.serial_control_container.show()
        # Horizontal box layout inside serial_control_container
        self.serial_layout = QHBoxLayout(self.serial_control_container)
        self.serial_layout.setContentsMargins(0, 0, 0, 0)
        self.serial_layout.setSpacing(20)
        # Serial Port ComboBox Label
        port_label = QLabel("Port:", self.serial_control_container)
        port_label.setFixedWidth(40)
        self.serial_layout.addWidget(port_label)
        # Serial Port ComboBox
        self.port_combobox = QComboBox(self.serial_control_container)
        self.refresh_serial_ports()
        self.serial_layout.addWidget(self.port_combobox)
        # Baudrate Label
        baud_label = QLabel("Baudrate:", self.serial_control_container)
        baud_label.setFixedWidth(60)
        self.serial_layout.addWidget(baud_label)

        # Baudrate ComboBox
        self.baud_combobox = QComboBox(self.serial_control_container)
        baudrates = ["9600", "19200", "38400", "57600", "115200"]
        self.baud_combobox.addItems(baudrates)
        self.baud_combobox.setCurrentText("9600")
        self.serial_layout.addWidget(self.baud_combobox)

        # Connect Button
        self.connect_btn = QPushButton("Connect", self.serial_control_container)
        self.connect_btn.clicked.connect(self.handle_connect_disconnect)
        self.serial_layout.addWidget(self.connect_btn)

         # Connection Status Indicator Label
        self.status_indicator = QLabel(self.serial_control_container)
        self.status_indicator.setFixedSize(20, 20)
        self.status_indicator.setStyleSheet("background-color: red; border: 1px solid black; border-radius: 10px;")
        self.serial_layout.addWidget(self.status_indicator)
        
        # Spacer to push items left
        self.serial_layout.addStretch()

        # Create label for "Vibration"
        self.vibration_label = QLabel("Vibration:", self)
        self.vibration_label.move(20, 500)  # Positioning the label
        self.vibration_label.resize(100, 125)  # Width, Height
        # Set font size for vibration label
        font_vibration = self.vibration_label.font()
        font_vibration.setPointSize(15)  # Set the desired font size
        font_vibration.setBold(True)
        self.vibration_label.setFont(font_vibration)
        self.vibration_label.show()

        # Create label for "angka"
        self.angka_label = QLabel("0", self)  # Initial value set to 0
        self.angka_label.move(160, 500)  # Positioning the label next to vibration label
        self.angka_label.resize(100, 125)  # Width, Height
        # Set font size for angka label
        font_angka = self.angka_label.font()
        font_angka.setBold(True)
        font_angka.setPointSize(15)  # Set the desired font size
        self.angka_label.setFont(font_angka)
        self.angka_label.show()

        # Create label for "Temperature"
        self.temperature_label = QLabel("Temperature:", self)
        self.temperature_label.move(20, 580)  # Positioning the label
        self.temperature_label.resize(200, 30)  # Width, Height
        # Set font size for temperature label
        font_temperature = self.temperature_label.font()
        font_temperature.setPointSize(15)  # Set the desired font size
        font_temperature.setBold(True)
        self.temperature_label.setFont(font_temperature)
        self.temperature_label.show()

         # Create label for "nilai temperature"
        self.nilai_temperature_label = QLabel("0", self)  # Initial value set to 0
        self.nilai_temperature_label.move(160, 580)  # Positioning the label next to vibration label
        self.nilai_temperature_label.resize(100, 30)  # Width, Height
        # Set font size for angka label
        font_angka_temperature = self.nilai_temperature_label.font()
        font_angka_temperature.setBold(True)
        font_angka_temperature.setPointSize(15)  # Set the desired font size
        self.nilai_temperature_label.setFont(font_angka_temperature)
        self.nilai_temperature_label.show()

        # Create label for "Kondisi motor"
        self.kondisiMotor_label = QLabel("Kondisi Motor:", self)
        self.kondisiMotor_label.move(350, 550)  # Positioning the label
        self.kondisiMotor_label.resize(200, 30)  # Width, Height
        # Set font size for kondisi motor label
        font_kondisiMotor = self.kondisiMotor_label.font()
        font_kondisiMotor.setPointSize(15)  # Set the desired font size
        font_kondisiMotor.setBold(True)
        self.kondisiMotor_label.setFont(font_kondisiMotor)
        self.kondisiMotor_label.show()

         # Create label for "nilai kondisi motor"
        self.nilai_kondisiMotor_label = QLabel("Normal", self)  # Initial value set to 0
        self.nilai_kondisiMotor_label.move(560, 490)  # Positioning the label next to vibration label
        self.nilai_kondisiMotor_label.resize(300, 150)  # Width, Height
        # Set font size for angka label
        font_angka_kondisiMotor = self.nilai_kondisiMotor_label.font()
        font_angka_kondisiMotor.setBold(True)
        font_angka_kondisiMotor.setPointSize(15)  # Set the desired font size
        self.nilai_kondisiMotor_label.setFont(font_angka_kondisiMotor)
        self.nilai_kondisiMotor_label.show()

        # Create label for "downtime motor"
        self.downtimeMotor_label = QLabel("Downtime Motor:", self)
        self.downtimeMotor_label.move(350, 580)  # Positioning the label
        self.downtimeMotor_label.resize(200, 30)  # Width, Height
        # Set font size for kondisi motor label
        font_downtimeMotor = self.downtimeMotor_label.font()
        font_downtimeMotor.setPointSize(15)  # Set the desired font size
        font_downtimeMotor.setBold(True)
        self.downtimeMotor_label.setFont(font_downtimeMotor)
        self.downtimeMotor_label.show()

         # Create label for "nilai downtime motor"
        self.nilai_downtime_label = QLabel("12.7", self)  # Initial value set to 0
        self.nilai_downtime_label.move(560, 580)  # Positioning the label next to vibration label
        self.nilai_downtime_label.resize(100, 30)  # Width, Height
        # Set font size for angka label
        font_angka_downtime = self.nilai_downtime_label.font()
        font_angka_downtime.setBold(True)
        font_angka_downtime.setPointSize(15)  # Set the desired font size
        self.nilai_downtime_label.setFont(font_angka_downtime)
        self.nilai_downtime_label.show()

        # Container widget for charts
        self.chart_container = QWidget(self)
        self.chart_container.setGeometry(110, 140, 960, 410)

        # Horizontal layout untuk dua grafik
        self.chart_layout = QHBoxLayout(self.chart_container)

        # === VIBRATION CHART ===
        self.vibration_x = []
        self.vibration_y = []
        self.vibration_z = []
        self.timestamps = []

        self.vibration_figure = Figure(figsize=(5, 2))
        self.vibration_canvas = FigureCanvas(self.vibration_figure)
        self.vibration_ax = self.vibration_figure.add_subplot(111)
        self.vibration_ax.set_title('Vibration')
        self.vibration_ax.set_xlabel('Time (s)')
        self.vibration_ax.set_ylabel('Vibration (mm/s²)')
        self.vibration_ax.grid(True)
        self.vibration_canvas.draw()

        # === SECOND CHART (e.g. Temperature) ===
        self.temp_data = []
        self.temp_time = []

        self.temp_figure = Figure(figsize=(5, 2))
        self.temp_canvas = FigureCanvas(self.temp_figure)
        self.temp_ax = self.temp_figure.add_subplot(111)
        self.temp_ax.set_title('Temperature')
        self.temp_ax.set_xlabel('Time (s)')
        self.temp_ax.set_ylabel('Temp (°C)')
        self.temp_ax.grid(True)
        self.temp_canvas.draw()

        # Tambahkan kedua canvas ke layout horizontal
        self.chart_layout.addWidget(self.vibration_canvas)
        self.chart_layout.addWidget(self.temp_canvas)
        self.chart_container.show() 

        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        if port_list:
            self.port_combobox.addItems(port_list)
            self.port_combobox.setCurrentIndex(0)
        else:
            self.port_combobox.addItem("No Ports")

        self.vibration_widgets = [
            self.section_title,
            self.serial_control_container,
            self.temperature_label,
            self.nilai_temperature_label,
            self.kondisiMotor_label,
            self.nilai_kondisiMotor_label,
            self.vibration_label,
            self.angka_label,
            self.downtimeMotor_label,
            self.nilai_downtime_label,
            self.vibration_canvas,
            self.temp_canvas
        ]

    def show_database(self):
        self.clear_content()

         # Create the label
        self.section_title = QLabel("DATABASE SECTION", self)
        # Manually position the label at x=50, y=30
        self.section_title.move(20, 70)
        # Optionally you can set fixed size or font to adjust appearance
        self.section_title.resize(200, 40)  # width, height
        font = self.section_title.font()
        font.setPointSize(10)
        font.setBold(True)
        self.section_title.setFont(font)
        self.section_title.show()
        
         # Create table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(20, 120, 900, 500)
        self.table_widget.setStyleSheet("QTableWidget { font-size: 10pt; }")
        
        # Add refresh button
        self.refresh_button = QPushButton("Refresh Data", self)
        self.refresh_button.setGeometry(750, 70, 150, 30)
        self.refresh_button.clicked.connect(self.load_database_data)
        
        # Add export button
        self.export_button = QPushButton("Export to CSV", self)
        self.export_button.setGeometry(580, 70, 150, 30)
        self.export_button.clicked.connect(self.export_to_csv)

        # Load initial data
        self.load_database_data()

        self.database_widgets = [
            self.section_title,
            self.table_widget,
            self.refresh_button,
            self.export_button
        ]

        for widget in self.database_widgets:
            widget.show()

    def load_database_data(self):
        """Load data from SQL Server database into the table"""
        try:
            conn = self.connect_to_db()
            if conn:
                cursor = conn.cursor()
                
                # Execute query to get data (adjust your table name)
                cursor.execute("SELECT * FROM tubes")
                rows = cursor.fetchall()
                
                # Get column names
                columns = [column[0] for column in cursor.description]
                
                # Set up table dimensions
                self.table_widget.setRowCount(len(rows))
                self.table_widget.setColumnCount(len(columns))
                self.table_widget.setHorizontalHeaderLabels(columns)
                
                # Populate table with data
                for row_idx, row in enumerate(rows):
                    for col_idx, item in enumerate(row):
                        self.table_widget.setItem(row_idx, col_idx, QTableWidgetItem(str(item)))
                
                # Adjust column widths
                self.table_widget.resizeColumnsToContents()
                
                # Add alternating row colors
                self.table_widget.setAlternatingRowColors(True)
                
                cursor.close()
                conn.close()
                
                # Show success in terminal/output if available
                if hasattr(self, 'terminal_output'):
                    self.terminal_output.append(f"Successfully loaded {len(rows)} rows from database")
                    
        except Exception as e:
            error_msg = f"Database error: {str(e)}"
            print(error_msg)
            if hasattr(self, 'terminal_output'):
                self.terminal_output.append(error_msg)

    def export_to_csv(self):
        """Export table data to CSV file"""
        try:
            docs_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            default_path = os.path.join(docs_dir, "data_export.csv")
            
            file_name, _ = QFileDialog.getSaveFileName(
                self, 
                "Save CSV File",
                default_path,
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_name:
                if not file_name.lower().endswith('.csv'):
                    file_name += '.csv'
                
                with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write headers
                    headers = []
                    for col in range(self.table_widget.columnCount()):
                        header = self.table_widget.horizontalHeaderItem(col)
                        headers.append(header.text() if header else "")
                    writer.writerow(headers)
                    
                    # Write data
                    for row in range(self.table_widget.rowCount()):
                        row_data = []
                        for col in range(self.table_widget.columnCount()):
                            item = self.table_widget.item(row, col)
                            row_data.append(item.text() if item else "")
                        writer.writerow(row_data)
                
                self.show_success_message(f"Data exported to:\n{os.path.abspath(file_name)}")
                
        except Exception as e:
            self.show_error_message(f"Export error: {str(e)}")

    def show_success_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Success")
        msg.setText(message)
        msg.exec()

    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.exec()

    def refresh_serial_ports(self):
        """Scan and populate available serial ports."""
        self.port_combobox.clear()
        ports = serial.tools.list_ports.comports()
        port_list = [port.device for port in ports]
        if port_list:
            self.port_combobox.addItems(port_list)
            self.port_combobox.setCurrentIndex(0)
        else:
            self.port_combobox.addItem("No Ports")

    def handle_connect_disconnect(self):
        if self.serial_port and self.serial_port.is_open:
            # Disconnect
            self.serial_port.close()
            self.serial_port = None
            self.connect_btn.setText("Connect")
            self.update_status_indicator(connected=False)
        else:
            # Connect
            selected_port = self.port_combobox.currentText()
            selected_baud = int(self.baud_combobox.currentText())
            if selected_port == "No Ports":
                QMessageBox.warning(self, "Warning", "No available serial ports detected.")
                return
            try:
                self.serial_port = serial.Serial(selected_port, selected_baud, timeout=1)
                self.connect_btn.setText("Disconnect")
                self.update_status_indicator(connected=True)
                self.start_reading_data()  # Start reading data after successful connection
            except (serial.SerialException, ValueError) as e:
                QMessageBox.critical(self, "Connection Error", f"Failed to connect to {selected_port} at {selected_baud} baud.\nError: {e}")

                self.serial_port = None
                self.update_status_indicator(connected=False)

    def update_status_indicator(self, connected: bool):
        if connected:
            color = "green"
        else:
            color = "red"
        self.status_indicator.setStyleSheet(f"background-color: {color}; border: 1px solid black; border-radius: 10px;")
    
    def start_reading_data(self):
        if self.serial_port and self.serial_port.is_open and not self.is_reading:
            self.timer.start(500)  # setiap 500 ms
            self.is_reading = True
    
    def setup_realtime_prediction(self):
        # Initialize models and scaler
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.load_models()
        
        # Initialize data buffers
        self.vibration_x = []
        self.vibration_y = []
        self.vibration_z = []
        self.temp_data = []
        self.timestamps = []
        self.temp_time = []

    def load_latest_models(self):
        """Load the latest trained models automatically from saved_models directory"""
        try:
            # Pastikan folder saved_models ada
            os.makedirs('saved_models', exist_ok=True)
            
            # ===== 1. LOAD CLASSIFIER =====
            classifier_files = glob.glob('saved_models/classifier_*.joblib')
            if classifier_files:
                # Ambil file terbaru berdasarkan timestamp
                latest_classifier = max(classifier_files, key=os.path.getctime)
                self.classifier_model = load(latest_classifier)
                print(f"Loaded classifier: {os.path.basename(latest_classifier)}")
            
            # ===== 2. LOAD REGRESSOR =====
            regressor_files = glob.glob('saved_models/regressor_*.joblib')
            if regressor_files:
                latest_regressor = max(regressor_files, key=os.path.getctime)
                self.regressor_model = load(latest_regressor)
                print(f"Loaded regressor: {os.path.basename(latest_regressor)}")
            
            # ===== 3. LOAD SCALER =====
            scaler_files = glob.glob('saved_models/scaler_*.joblib')
            if scaler_files:
                latest_scaler = max(scaler_files, key=os.path.getctime)
                self.scaler = load(latest_scaler)
                print(f"Loaded scaler: {os.path.basename(latest_scaler)}")
            
            # ===== 4. VERIFIKASI =====
            loaded_models = []
            if hasattr(self, 'classifier_model'):
                loaded_models.append("Classifier")
            if hasattr(self, 'regressor_model'):
                loaded_models.append("Regressor")
            if hasattr(self, 'scaler'):
                loaded_models.append("Scaler")
                
            if loaded_models:
                QMessageBox.information(self, "Sukses", 
                                    f"Model berhasil dimuat:\n{', '.join(loaded_models)}")
            else:
                QMessageBox.warning(self, "Peringatan", 
                                "Tidak ada model yang ditemukan di folder saved_models")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal memuat model: {str(e)}")

    def read_data(self):
        """Read data from serial port and make real-time predictions"""
        if self.serial_port and self.serial_port.in_waiting > 0:
            try:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').rstrip()
                values = line.split('/')
                
                if len(values) >= 5:
                    # Parse sensor values
                    try:
                        x = float(values[0])
                        y = float(values[1])
                        z = float(values[2])
                        vibration = float(values[3])
                        temp = float(values[4])
                    except ValueError as ve:
                        print(f"ValueError while parsing sensor data: {ve}")
                        return

                    # Update sensor value displays
                    self.angka_label.setText(f"{vibration:.2f}")
                    self.nilai_temperature_label.setText(f"{temp:.2f}")

                    condition_pred = None
                    downtime_pred = None

                    # Periksa model dengan nama atribut yang benar
                    if all([hasattr(self, 'classifier_model'), 
                        hasattr(self, 'regressor_model'), 
                        hasattr(self, 'scaler')]):
                        
                        features = np.array([[temp, vibration]])  # Pastikan urutan fitur sama dengan training
                        print(f"Input features: {features}")
                        
                        try:
                            scaled_features = self.scaler.transform(features)

                            # Classification prediction
                            condition_pred = self.classifier_model.predict(scaled_features)[0]
                            condition_prob = np.max(self.classifier_model.predict_proba(scaled_features))

                            # Regression prediction
                            downtime_pred = self.regressor_model.predict(scaled_features)[0]

                            # Update UI
                            self.update_prediction_ui(condition_pred, condition_prob, downtime_pred)

                        except Exception as e:
                            print(f"Prediction error: {e}")

                    # Store data for plotting
                    self.store_plot_data(x, y, z, temp)

                    # Save to database
                    if condition_pred is not None and downtime_pred is not None:
                        self.save_to_database(temp, vibration, condition_pred, downtime_pred)

            except Exception as e:
                print(f"Error processing serial data: {e}")
                # Tambahkan logging error ke UI jika perlu
                self.terminal_output.append(f"Serial Error: {str(e)}")

    def update_prediction_ui(self, condition_code, probability, downtime):
        """Update the prediction labels with color coding"""

        # Handle non-int condition inputs safely
        try:
            condition_code = int(condition_code)
        except ValueError:
            condition_code = -1

        # Get class info (label, color, description)
        class_info = self.class_mapping.get(condition_code, {
            "label": f"Unknown ({condition_code})", 
            "color": "black",
            "description": "Status tidak dikenali"
        })

        label = class_info["label"]
        color = class_info["color"]

        # Update condition label
        self.nilai_kondisiMotor_label.setText(label)
        self.nilai_kondisiMotor_label.setStyleSheet(f"color: {color};")

        # Update downtime label
        self.nilai_downtime_label.setText(f"{downtime:.2f}")
        
        # Color code downtime based on severity
        if downtime > 15:
            self.nilai_downtime_label.setStyleSheet("color: red;")
        elif downtime > 8:
            self.nilai_downtime_label.setStyleSheet("color: orange;")
        else:
            self.nilai_downtime_label.setStyleSheet("color: green;")

    def store_plot_data(self, x, y, z, temp):
        """Store sensor data for plotting"""
        self.vibration_x.append(x)
        self.vibration_y.append(y)
        self.vibration_z.append(z)
        self.temp_data.append(temp)
        self.timestamps.append(len(self.timestamps))
        
        # Limit data points
        max_points = 1000
        if len(self.timestamps) > max_points:
            self.vibration_x = self.vibration_x[-max_points:]
            self.vibration_y = self.vibration_y[-max_points:]
            self.vibration_z = self.vibration_z[-max_points:]
            self.temp_data = self.temp_data[-max_points:]
            self.timestamps = self.timestamps[-max_points:]
        
        # Update plots
        self.update_plots()

    def update_plots(self):
        """Update vibration and temperature plots"""
        # Update vibration plot
        self.vibration_ax.clear()
        self.vibration_ax.plot(self.timestamps, self.vibration_x, label='X', color='r')
        self.vibration_ax.plot(self.timestamps, self.vibration_y, label='Y', color='g')
        self.vibration_ax.plot(self.timestamps, self.vibration_z, label='Z', color='b')
        self.vibration_ax.set_title('Vibration Sensor Data')
        self.vibration_ax.legend()
        self.vibration_canvas.draw()
        
        # Update temperature plot
        self.temp_ax.clear()
        self.temp_ax.plot(self.timestamps, self.temp_data, label='Temperature', color='m')
        self.temp_ax.set_title('Temperature Sensor Data')
        self.temp_ax.legend()
        self.temp_canvas.draw()

    def save_to_database(self, temp, vibration, condition, downtime):
        """Save data and predictions to database"""
        try:
            conn = self.connect_to_db()
            if conn:
                cursor = conn.cursor()
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if label_kode column exists
                cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.columns 
                            WHERE object_id = OBJECT_ID('tubes') AND name = 'label_kode')
                BEGIN
                    ALTER TABLE tubes ADD label_kode VARCHAR(10) NULL
                END
                """)
                
                # Insert data
                query = """
                INSERT INTO tubes (timestamp, temperature, vibration, 
                                kondisi_motor, downtime_motor, label_kode)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                # Get label code based on condition
                label_code = self.get_label_code(condition)
                cursor.execute(query, (timestamp, temp, vibration, 
                                    condition, downtime, label_code))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Database save error: {e}")

    def get_label_code(self, condition_code):
        """Map condition code to label code"""
        try:
            condition_code = int(condition_code)
        except ValueError:
            return 'UNK'

        label = self.class_mapping.get(condition_code, {}).get("label", "").lower()

        code_map = {
            'normal': 'C5',
            'perlu perawatan': 'C4',
            'indikasi rusak': 'C3',
            'getaran bahaya': 'C3',
            'suhu kritis, getaran bahaya': 'C1',
            'suhu tinggi, getaran rendah': 'C2',
            'suhu rendah, getaran tinggi': 'C2',
            'suhu rendah': 'C4'
        }

        return code_map.get(label, 'UNK')
        
    def update_section(self, name):
        self.section_title.setText(f"{name.upper()} SECTION")
        self.clear_content()

    def show_notification_window(self):
    
        # Siapkan pesan awal
        message = ""

        # Cek apakah file classifier dan regressor tersedia
        if hasattr(self, 'classifier_model_path') and Path(self.classifier_model_path).exists():
            classifier = joblib.load(self.classifier_model_path)
            message += "[Random Forest Classifier]\n"
            message += f"Tipe objek: {type(classifier)}\n"
            message += f"Jumlah estimators: {len(classifier.estimators_)}\n"
            message += f"Max depth: {classifier.max_depth}\n"
            message += f"Feature importances: {classifier.feature_importances_}\n"
            message += "Parameter model:\n"
            for k, v in classifier.get_params().items():
                message += f"  {k}: {v}\n"
            message += "\n"

        if hasattr(self, 'regressor_model_path') and Path(self.regressor_model_path).exists():
            regressor = joblib.load(self.regressor_model_path)
            message += "[Random Forest Regressor]\n"
            message += f"Tipe objek: {type(regressor)}\n"
            message += f"Jumlah estimators: {len(regressor.estimators_)}\n"
            message += f"Max depth: {regressor.max_depth}\n"
            message += f"Feature importances: {regressor.feature_importances_}\n"
            message += "Parameter model:\n"
            for k, v in regressor.get_params().items():
                message += f"  {k}: {v}\n"
            message += "\n"

        # Tampilkan pesan
        msg = QMessageBox(self)
        msg.setWindowTitle("Training Model Selesai")
        msg.setIcon(QMessageBox.Icon.Information)

        # Batasi isi jika terlalu panjang
        if len(message) > 2000:
            message = message[:2000] + "\n... (dipotong)"

        msg.setText("Model training selesai.\nKlik 'Detail' untuk informasi model.")
        msg.setDetailedText(message)
        msg.exec()
    
    def clear_terminal_output(self):
        self.terminal_output.clear()
    
    def clear_terminal_output2(self):
        self.terminal_output2.clear()

    def show_training_option_dialog(self):
        """Menampilkan dialog konfirmasi training dengan validasi input"""
        # Cek kelengkapan konfigurasi
        if not self.is_model_config_complete():
            QMessageBox.critical(
                self,
                "Konfigurasi Tidak Lengkap",
                "Lengkapi dulu semua parameter training:\n"
                "- Pilih preprocessing\n"
                "- Pilih jenis model\n"
                "- Isi n_estimator dan max_depth",
                QMessageBox.StandardButton.Ok
            )
            self.terminal_output.append("ERROR: Konfigurasi model belum lengkap")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Konfirmasi Training Model")

        layout = QVBoxLayout()

        label = QLabel("Apakah Anda ingin memulai training model?")
        layout.addWidget(label)

        # Tampilkan ringkasan konfigurasi
        config_summary = self.get_config_summary()
        summary_label = QLabel(config_summary)
        summary_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(summary_label)

        # Tombol OK dan Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec():
            self.reset_ui_elements()
            self.terminal_output.append("Memulai training model dengan konfigurasi:")
            self.terminal_output.append(config_summary)
            self.train_model()
        else:
            self.terminal_output.append("Training model dibatalkan.")

    def is_model_config_complete(self):
        """Validasi kelengkapan konfigurasi model"""
        # Cek preprocessing terpilih
        if not (self.checkbox_minmax.isChecked() or self.checkbox_standard.isChecked()):
            return False
        
        # Cek jenis model terpilih
        if not (self.checkbox_rf_classifier.isChecked() or self.checkbox_rf_regressor.isChecked()):
            return False
        
        # Cek n_estimator dan max_depth
        if not self.n_estimators_input.text() or not self.max_depth_input.text():
            return False
        
        return True

    def get_config_summary(self):
        """Membuat ringkasan konfigurasi dalam format teks"""
        preprocessing = "MinMaxScaler" if self.checkbox_minmax.isChecked() else "StandardScaler" if self.checkbox_standard.isChecked() else "-"
        model_type = "Classifier" if self.checkbox_rf_classifier.isChecked() else "Regressor"
        criterion = self.criterion_combo.currentText() if self.checkbox_rf_classifier.isChecked() else self.metric_combo.currentText()
        
        summary = (
            "Konfigurasi Model:\n"
            f"Preprocessing: {preprocessing}\n"
            f"Jenis Model: Random Forest {model_type}\n"
            f"n_estimators: {self.n_estimators_input.text()}\n"
            f"max_depth: {self.max_depth_input.text()}\n"
            f"Criterion/Metric: {criterion}"
        )
        return summary

if __name__ == "__main__":
    app =  QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
