import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import os
import threading

root = tk.Tk()
root.title("Trading Bot")
root.geometry("650x600")
root.configure(bg="bisque")

# === STATE VARS ===
mode = tk.StringVar(value="live")
csv_file_name = tk.StringVar()

# Load defaults from settings
with open('settings.txt', 'r') as file:
    orders = float(file.readline().strip())
    risk = float(file.readline().strip())

# === UTILITY ===
def validate_float(d):
    return d == "" or d.replace(".", "", 1).isdigit()

vflt = root.register(validate_float)

# === HEADING ===
ttk.Label(root, text="Welcome to Trading Bot", font=("Cambria", 20, "bold")).pack(pady=10)
ttk.Label(root, text="This trading bot trades EURUSD using MetaTrader 5 or Offline ML").pack()
ttk.Label(root, text=f"Max Order Size: ${orders}, Risk: {risk}%").pack()
ttk.Label(root, text="Please enter the information below").pack()

# === ORDER ENTRY ===
order_label = ttk.Label(root, text="Maximum Order Size ($):")
order_label.pack()
order_entry = ttk.Entry(root, validate="key", validatecommand=(vflt, "%P"))
order_entry.pack()

# === RISK ENTRY ===
risk_label = ttk.Label(root, text="Maximum Risk (%):")
risk_label.pack()
risk_entry = ttk.Entry(root, validate="key", validatecommand=(vflt, "%P"))
risk_entry.pack()

# === CSV ENTRY ===
csv_label = ttk.Label(root, text="CSV File (Offline Mode Only):")
csv_label.pack()
csv_frame = tk.Frame(root, bg="bisque")
csv_frame.pack()

csv_entry = ttk.Entry(csv_frame, textvariable=csv_file_name, width=40)
csv_entry.pack(side=tk.LEFT, padx=5)

def upload_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        csv_file_name.set(os.path.basename(file_path))
        messagebox.showinfo("CSV Uploaded", f"Using file: {csv_file_name.get()}")

upload_button = ttk.Button(csv_frame, text="+", width=3, command=upload_csv)
upload_button.pack(side=tk.LEFT)

# === MODE SELECTOR ===
def mode_changed():
    is_offline = mode.get() == "offline"
    if is_offline:
        messagebox.showinfo("Offline Mode", "Offline Mode is only used for training the machine learning.")
        order_entry.config(state="disabled")
        risk_entry.config(state="disabled")
        log_frame.pack(pady=10)
    else:
        order_entry.config(state="normal")
        risk_entry.config(state="normal")
        csv_file_name.set("")
        log_frame.pack_forget()

mode_frame = ttk.LabelFrame(root, text="Mode Selection")
mode_frame.pack(pady=10)

offline_radio = ttk.Radiobutton(mode_frame, text="Offline Mode", variable=mode, value="offline", command=mode_changed)
offline_radio.grid(row=0, column=0, padx=10)

live_radio = ttk.Radiobutton(mode_frame, text="Live Mode", variable=mode, value="live", command=mode_changed)
live_radio.grid(row=0, column=1, padx=10)

# === LOG DISPLAY BOX ===
log_frame = tk.Frame(root, bg="bisque")
log_box = scrolledtext.ScrolledText(log_frame, height=10, width=75, state='disabled')
log_box.pack()

def log_output(msg):
    log_box.config(state='normal')
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    log_box.config(state='disabled')

# === SUBMIT FUNCTION ===
def submit():
    global orders, risk

    if mode.get() == "offline":
        if not csv_file_name.get():
            messagebox.showerror("Missing CSV", "Please upload or enter a CSV file name for offline training.")
            return

        with open('settings.txt', 'w') as file:
            file.write("0\n")
            file.write("0\n")
            file.write(f"{csv_file_name.get()}\n")

        run_training_script()

    else:
        order_val = order_entry.get()
        risk_val = risk_entry.get()

        if order_val:
            try:
                order_float = float(order_val)
                if order_float > 100:
                    messagebox.showerror("Error", "Order size too big! Max is $100.")
                    return
                existing_orders = order_val
                existing_risk = "0"
            except ValueError:
                messagebox.showerror("Error", "Invalid order input.")
                return
        elif risk_val:
            try:
                risk_float = float(risk_val)
                if risk_float > 0.3:
                    messagebox.showerror("Error", "Risk too high! Max is 0.3%.")
                    return
                existing_risk = risk_val
                existing_orders = "0"
            except ValueError:
                messagebox.showerror("Error", "Invalid risk input.")
                return
        else:
            messagebox.showerror("Error", "Please enter either Order or Risk.")
            return

        with open('settings.txt', 'w') as file:
            file.write(f'{existing_orders}\n')
            file.write(f'{existing_risk}\n')

        messagebox.showinfo("Live Bot", "Launching live trading bot...")
        subprocess.Popen(["python", "message_live.py"])

# === RUN TRAINING SCRIPT W/ OUTPUT CONTROL ===
def run_training_script():
    def target():
        display_log = messagebox.askyesno("Training Output", "Would you like to display the training logs?")
        process = subprocess.Popen(["python", "message.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if display_log:
            for line in iter(process.stdout.readline, ''):
                log_output(line.strip())
        else:
            for _ in iter(process.stdout.readline, ''):
                pass
        process.stdout.close()
        process.wait()
        messagebox.showinfo("Training Complete", "The model was trained successfully!")

    thread = threading.Thread(target=target)
    thread.start()

# === BUTTONS ===
submit_button = ttk.Button(root, text="Train", command=submit)
submit_button.pack(pady=10)

exit_button = ttk.Button(root, text="EXIT", command=root.destroy)
exit_button.pack(pady=5)

# === START ===
mode_changed()  # initialize mode state
root.mainloop()
