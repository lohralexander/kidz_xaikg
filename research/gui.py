import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functions import *
from research.research_config import Initialization
import tkinter as tk
from tkinter import messagebox


# Function to be called when the button is pressed
def on_start():
    param1 = entry1.get()
    param2 = entry2.get()
    param3 = entry3.get()

    # Convert input to appropriate types if needed (e.g., int, float)
    try:
        param1 = int(param1)
        param2 = int(param2)
        param3 = int(param3)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")
        return
    init_values = Initialization()
    # Call your function with the parameters
    result = start_research_run(init_values.get_ontology(),init_values.get_questionnaire(), param2, param3)

    # Display result in a messagebox or update the GUI with the result
    messagebox.showinfo("Result", f"Function Result: {result}")


def main():

    # Create the main window
    root = tk.Tk()
    root.title("Parameter Input GUI")

    # Create labels and entry widgets for parameters
    label1 = tk.Label(root, text="Parameter 1 (integer):")
    label1.grid(row=0, column=0, padx=10, pady=10)

    entry1 = tk.Entry(root)
    entry1.grid(row=0, column=1, padx=10, pady=10)

    label2 = tk.Label(root, text="Parameter 2 (integer):")
    label2.grid(row=1, column=0, padx=10, pady=10)

    entry2 = tk.Entry(root)
    entry2.grid(row=1, column=1, padx=10, pady=10)

    label3 = tk.Label(root, text="Parameter 3 (integer):")
    label3.grid(row=1, column=0, padx=10, pady=10)

    entry3 = tk.Entry(root)
    entry3.grid(row=1, column=1, padx=10, pady=10)

    # Create a button that calls the on_start function when pressed
    start_button = tk.Button(root, text="Start", command=on_start)
    start_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == '__main__':
    main()