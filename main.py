import tkinter as tk
from gui import App

def main():
    while True:
        root = tk.Tk()
        app = App(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()


if __name__ == "__main__":
    main()