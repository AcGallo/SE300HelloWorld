import tkinter as tk
from tkinter import messagebox

class TradingGUI:
    def __init__(self, trading_instance):
        self.trading = trading_instance
        self.root = tk.Tk()
        self.root.title("MT5 Trading GUI")

        self.label = tk.Label(self.root, text="Enter Volume:")
        self.label.pack()
        
        self.volume_entry = tk.Entry(self.root)
        self.volume_entry.pack()
        
        self.buy_button = tk.Button(self.root, text="Buy", command=lambda: self.execute_trade("buy"))
        self.buy_button.pack()
        
        self.sell_button = tk.Button(self.root, text="Sell", command=lambda: self.execute_trade("sell"))
        self.sell_button.pack()
        
        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit)
        self.exit_button.pack()
    
    def execute_trade(self, order_type):
        try:
            volume = float(self.volume_entry.get())
            message = self.trading.place_order(order_type, volume)
            messagebox.showinfo("Trade Status", message)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for volume.")
    
    def run(self):
        self.root.mainloop()
