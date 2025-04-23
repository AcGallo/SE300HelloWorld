import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess  

def validate_float(d):
    if d == "" or d.replace(".", "", 1).isdigit():  
        return True
    return False

def clear_other_entries(active_entry):
    if active_entry == order_entry:
        risk_entry.delete(0, tk.END)
    elif active_entry == risk_entry:
        order_entry.delete(0, tk.END)

def submit():
    global orders
    global risk

    max_order = 10000  # max allowed order value
    max_risk = 50      # max allowed risk percentage

    order_val = order_entry.get()
    risk_val = risk_entry.get()

    if order_val:
        try:
            order_float = float(order_val)
            if order_float > max_order:
                messagebox.showerror("Error", "Order size too big! Max allowed is $10,000.")
                return
            existing_orders = order_val
            existing_risk = "0"
        except ValueError:
            messagebox.showerror("Error", "Invalid order input.")
            return
    elif risk_val:
        try:
            risk_float = float(risk_val)
            if risk_float > max_risk:
                messagebox.showerror("Error", "Risk too big! Max allowed is 50%.")
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

    print(f"Saved Orders: {existing_orders}, Risk: {existing_risk}%")
    messagebox.showinfo("Success", "Submitted successfully!")
    subprocess.Popen(["python", "mt5_trading.py"])

global orders
global risk

with open('settings.txt', 'r') as file:
    orders = float(file.readline().strip())
    risk = float(file.readline().strip())

root = tk.Tk()
root.title("Trading Bot")
root.geometry("600x400")
root.configure(bg="bisque")

vflt = root.register(validate_float)

welcome_label = ttk.Label(root, text="Welcome to Trading Bot", font=("Cambria", 20, "bold"))
welcome_label.pack(pady=10)

information_label = ttk.Label(root, text="This trading bot trades EURUSD stock only using Meta Trader 5 API")
information_label.pack()

current_value_label = ttk.Label(root, text=f"Max Order Size is currently set to ${orders} and risk percent set to {risk}")
current_value_label.pack()

instruction_label = ttk.Label(root, text="Please enter the information below")
instruction_label.pack()

order_label = ttk.Label(root, text="Maximum Order Size: $")
order_label.pack()
order_entry = ttk.Entry(root, validate="key", validatecommand=(vflt, "%P"))
order_entry.pack()

risk_label = ttk.Label(root, text="Maximum Risk: %")
risk_label.pack()
risk_entry = ttk.Entry(root, validate="key", validatecommand=(vflt, "%P"))
risk_entry.pack()

order_entry.bind("<Key>", lambda event: clear_other_entries(order_entry))
risk_entry.bind("<Key>", lambda event: clear_other_entries(risk_entry))

submit_button = ttk.Button(root, text="SUBMIT", command=submit)
submit_button.pack(pady=10)

exit_button = ttk.Button(root, text="EXIT", command=root.destroy)
exit_button.pack(pady=10)

root.mainloop()
