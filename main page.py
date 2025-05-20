import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import labeling
import serial.tools.list_ports
import serial
import threading
import time
import pyodbc

def connect_to_database():
    try:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=LAPTOP-4Q9VNCT5;'  # Ganti sesuai nama servermu
            'DATABASE=tubesAI;'         # Ganti sesuai nama databasenya
            'UID=sa;'                   # Ganti sesuai user
            'PWD=Rifkiap18'             # Ganti sesuai password
        )
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 5 * FROM tubes")
        rows = cursor.fetchall()
        print("Data from database:")
        for row in rows:
            print(row)
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error connecting to database:", e)

# You can call connect_to_database() function where appropriate in your code
# For example, call it here to test connection when running this script directly
if __name__ == "__main__":
    connect_to_database()
   
def main():
    root = tk.Tk()
    root.title("Tkinter Window")
    root.geometry("1080x620")  # Width x Height

   # === Top Frame utama yang berisi seluruh header ===
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=0)

    # === Frame horizontal untuk baris atas (judul dan port) ===
    top_bar_frame = tk.Frame(top_frame)
    top_bar_frame.pack(fill=tk.X, padx=10)

    # Frame kiri untuk judul
    title_frame = tk.Frame(top_bar_frame)
    title_frame.pack(side=tk.LEFT, anchor='w')

    # Frame kanan untuk port dan baudrate
    comms_frame = tk.Frame(top_bar_frame)
    comms_frame.pack(side=tk.RIGHT, anchor='e')

    # === Judul aplikasi ===
    app_title = tk.Label(title_frame, text="PREDICTIVE MAINTENANCE", font=("Arial", 16, "bold"))
    app_title.pack(anchor='w')

    # === Combobox Port ===
    ports = [port.device for port in serial.tools.list_ports.comports()]
    port_combobox = ttk.Combobox(comms_frame, values=ports, state="readonly", width=15)
    port_combobox.pack(side=tk.LEFT, padx=5)
    if ports:
        port_combobox.current(0)

    # === Combobox Baudrate ===
    baudrate_values = ["9600", "19200", "38400", "57600", "115200"]
    baudrate_combobox = ttk.Combobox(comms_frame, values=baudrate_values, state="readonly", width=7)
    baudrate_combobox.pack(side=tk.LEFT, padx=5)
    baudrate_combobox.current(0)

    # === LED Indicator ===
    led_canvas = tk.Canvas(comms_frame, width=20, height=20, highlightthickness=0)
    led_canvas.pack(side=tk.LEFT, padx=(0, 10))

    # === Frame untuk Database dan Tabel (di bawah judul) ===
    info_frame = tk.Frame(top_frame)
    info_frame.pack(anchor='w', padx=10, pady=(5, 0))

    # === Database Frame ===
    database_frame = tk.Frame(info_frame)
    database_label = tk.Label(database_frame, text="Database", font=("Arial", 12))
    database_label.pack(side=tk.LEFT)
    database_box_frame = tk.Frame(database_frame, width=200, height=25, bd=1, relief=tk.SUNKEN)
    database_box_frame.pack(side=tk.LEFT, padx=(20, 10))
    database_frame.pack(anchor='w', pady=3)
    database_frame.pack_forget()  # Hide by default

    # === Tabel Frame ===
    tabel_frame = tk.Frame(info_frame)
    tabel_label = tk.Label(tabel_frame, text="Tabel", font=("Arial", 12))
    tabel_label.pack(side=tk.LEFT)
    tabel_box_frame = tk.Frame(tabel_frame, width=200, height=25, bd=1, relief=tk.SUNKEN)
    tabel_box_frame.pack(side=tk.LEFT, padx=(20, 10))
    tabel_frame.pack(anchor='w', pady=5)
    tabel_frame.pack_forget()  # Hide by default

    # Function to update LED color based on port connection status
    def update_led():
        selected_port = port_combobox.get()
        connected_ports = [port.device for port in serial.tools.list_ports.comports()]
        led_canvas.delete("all")
        if selected_port in connected_ports:
            led_canvas.create_oval(2, 2, 18, 18, fill="green", outline="black")
        else:
            led_canvas.create_oval(2, 2, 18, 18, fill="red", outline="black")

    # Bind combobox selection change to update LED
    def on_port_selected(event):
        update_led()

    port_combobox.bind("<<ComboboxSelected>>", on_port_selected)

    # Initial LED update
    update_led()

    # Create menu bar
    menubar = tk.Menu(root)
    root.config(menu=menubar)

    # Add menu items
    menubar.add_command(label="Vibration")
    menubar.add_command(label="Temperature")
    menubar.add_command(label="Current")
    menubar.add_command(label="Training", command=lambda: on_training_menu())
    menubar.add_command(label="Evaluasi", command=lambda: on_evaluasi_menu())

    # Highlight "Vibration" menu item to indicate current page
    menubar.entryconfig("Vibration", background="blue", foreground="white")

    # Create a matplotlib figure
    # Change the figsize tuple (width, height) in inches to adjust the size of the line chart
    # For example, figsize=(5, 4) means width=5 inches, height=4 inches
    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Embed the matplotlib figure in Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, pady=5)

   # Data lists for each axis
    x_data = []            # common x-axis (e.g. timestamps or count)
    x_accel = []           # acceleration X
    y_accel = []           # acceleration Y
    z_accel = []           # acceleration Z

    def update_chart(title, x_data, y_datas, y_labels):
        ax.clear()
        colors = ['r', 'g', 'b']
        for i, y_data in enumerate(y_datas):
            ax.plot(x_data, y_data, marker='o', linestyle='-', color=colors[i % len(colors)], label=y_labels[i])
        ax.set_title(title)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.legend()
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        canvas.draw()

    def on_temperature_menu():
        x_temp = [0, 1, 2, 3, 4, 5]
        y_temp = [5, 3, 6, 2, 7, 4]
        update_chart("Temperature Chart", x_temp, y_temp)
        menubar.entryconfig("Vibration", background="SystemMenu")
        menubar.entryconfig("Temperature", background="blue", foreground="white")
        menubar.entryconfig("Current", background="SystemMenu")

        # Show chart and indicators
        canvas.get_tk_widget().pack(side=tk.TOP, pady=20)
        main_indicator_frame.pack(side=tk.TOP, pady=(10, 20), anchor='w')
        downtime_frame.pack(side=tk.TOP, pady=(0, 20), anchor='w')

        # Show vibration and kondisiMotor frames
        vibration_frame.pack(side=tk.LEFT, padx=(20, 0))
        kondisiMotor_frame.pack(side=tk.LEFT)

        # Hide database and tabel frames
        database_frame.pack_forget()
        tabel_frame.pack_forget()

    # Fungsi panggilan dari menu "Vibration"
    def on_vibration_menu():
        update_chart(
            "Vibration Motor Graph",
            x_data,
            [x_accel, y_accel, z_accel],
            ["Accel X", "Accel Y", "Accel Z"]
        )
       # Update warna menu aktif
        menubar.entryconfig("Vibration", background="blue", foreground="white")
        menubar.entryconfig("Temperature", background="SystemMenu")
        menubar.entryconfig("Current", background="SystemMenu")
        menubar.entryconfig("Training", background="SystemMenu")
        menubar.entryconfig("Evaluasi", background="SystemMenu")

        # Tampilkan chart dengan jarak minimal dari judul
        canvas.get_tk_widget().pack_forget()  # Hapus posisi sebelumnya
        canvas.get_tk_widget().pack(side=tk.TOP, pady=(0, 0))  # Minimal padding
        
        # Tampilkan indikator dan info lainnya
        main_indicator_frame.pack_forget()
        main_indicator_frame.pack(side=tk.TOP, pady=(0, 0), anchor='w')

        kondisiMotor_frame.pack_forget()
        kondisiMotor_frame.pack(side=tk.TOP, anchor='w', pady=(2, 0))

        downtime_frame.pack_forget()
        downtime_frame.pack(side=tk.TOP, pady=(2, 0), anchor='w')

        vibration_frame.pack_forget()
        vibration_frame.pack(side=tk.LEFT, padx=(20, 0), anchor='n')

        # Sembunyikan frame training
        database_frame.pack_forget()
        tabel_frame.pack_forget()

    def on_current_menu():
        x_current = [0, 1, 2, 3, 4, 5]
        y_current = [2, 4, 1, 3, 5, 2]
        update_chart("Current Chart", x_current, y_current)
        menubar.entryconfig("Vibration", background="SystemMenu")
        menubar.entryconfig("Temperature", background="SystemMenu")
        menubar.entryconfig("Current", background="blue", foreground="white")

        # Show chart and indicators
        canvas.get_tk_widget().pack(side=tk.TOP, pady=20)
        main_indicator_frame.pack(side=tk.TOP, pady=(10, 20), anchor='w')
        downtime_frame.pack(side=tk.TOP, pady=(0, 20), anchor='w')

        # Show vibration and kondisiMotor frames
        vibration_frame.pack(side=tk.LEFT, padx=(20, 0))
        kondisiMotor_frame.pack(side=tk.LEFT)

        # Hide database and tabel frames
        database_frame.pack_forget()
        tabel_frame.pack_forget()

    def clear_chart_and_indicators(title=""):
        # Kosongkan chart
        hide_main_widgets()
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        canvas.draw()
        vibration_value_var.set("0.00")

    def hide_main_widgets():
        canvas.get_tk_widget().pack_forget()
        main_indicator_frame.pack_forget()
        downtime_frame.pack_forget()

    # Create a new main container frame below the chart to hold all three indicators
    main_indicator_frame = tk.Frame(root)
    main_indicator_frame.pack(side=tk.TOP, pady=(10, 20), anchor='w')

    # Create a left vertical frame inside main_indicator_frame to stack disiMotor and downtime vertically
    left_vertical_frame = tk.Frame(main_indicator_frame)
    left_vertical_frame.pack(side=tk.LEFT, anchor='n')

    # Add kondisiMotor_frame inside left_vertical_frame
    kondisiMotor_frame = tk.Frame(left_vertical_frame)
    kondisiMotor_frame.pack(side=tk.TOP, anchor='w')

    kondisiMotor_label = tk.Label(kondisiMotor_frame, text="Kondisi motor", font=("Arial", 12))
    kondisiMotor_label.pack(side=tk.LEFT)

    # Add a box (Frame widget) next to the label
    kondisiMotor_box_frame = tk.Frame(kondisiMotor_frame, width=200, height=25, bd=1, relief=tk.SUNKEN)
    kondisiMotor_box_frame.pack(side=tk.LEFT, padx=(10, 10), anchor='w')

    # Add downtime_frame inside left_vertical_frame below kondisiMotor_frame
    downtime_frame = tk.Frame(left_vertical_frame)
    downtime_frame.pack(side=tk.TOP, pady=(10, 0), anchor='w')

    downtime_label = tk.Label(downtime_frame, text="Downtime Motor", font=("Arial", 12))
    downtime_label.pack(side=tk.LEFT)

    # Add a frame next to the Downtime Motor label (for future widgets)
    downtime_box_frame = tk.Frame(downtime_frame, width=200, height=25, bd=1, relief=tk.SUNKEN)
    downtime_box_frame.pack(side=tk.LEFT, padx=(10, 10), anchor='w')

    # Add vibration_frame inside main_indicator_frame to the right of left_vertical_frame
    vibration_frame = tk.Frame(main_indicator_frame)
    vibration_frame.pack(side=tk.LEFT, padx=(20, 0), anchor='n')

    vibration_label = tk.Label(vibration_frame, text="Vibration", font=("Arial", 12))
    vibration_label.pack(side=tk.LEFT)

    vibration_box_frame = tk.Frame(vibration_frame, width=200, height=25, bd=1, relief=tk.SUNKEN)
    vibration_box_frame.pack(side=tk.LEFT, padx=(10, 10))

    # StringVar agar mudah diupdate
    vibration_value_var = tk.StringVar(value="0.00")
    vibration_value_label = tk.Label(vibration_box_frame, textvariable=vibration_value_var, font=("Arial", 12))
    vibration_value_label.pack(expand=True)

    def on_training_menu():
        clear_chart_and_indicators("Training Model")
        menubar.entryconfig("Training", background="blue", foreground="white")
        menubar.entryconfig("Evaluasi", background="SystemMenu", foreground="black")

        # Show database and tabel indicator frames
        database_frame.pack(side=tk.TOP, anchor='w', pady=(5, 0))
        tabel_frame.pack(side=tk.TOP, anchor='w', pady=(5, 0))
        # Hide other indicator frames
        kondisiMotor_frame.pack_forget()
        vibration_frame.pack_forget()

         # Ensure top frame with app title, port combobox, and baudrate combobox is visible
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=7)

    def on_evaluasi_menu():
        clear_chart_and_indicators("Evaluasi Model")
        menubar.entryconfig("Evaluasi", background="blue", foreground="white")
        menubar.entryconfig("Training", background="SystemMenu", foreground="black")

    menubar.entryconfig("Temperature", command=on_temperature_menu)
    menubar.entryconfig("Vibration", command=on_vibration_menu)
    menubar.entryconfig("Current", command=on_current_menu)

    # Call on_vibration_menu to display vibration menu initially
    on_vibration_menu()

    root.mainloop()

if __name__ == "__main__":
    main()