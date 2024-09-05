import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functions import *


class Gui:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Research Toolkit")

        # Create dropdowns for parameters
        self.param1_label = ttk.Label(root, text="Demo Mode:")
        self.param1_label.pack(pady=5)
        self.param1 = ttk.Combobox(root, values=[True, False])
        self.param1.pack(pady=5)
        self.param1.current(0)

        self.param2_label = ttk.Label(root, text="Search Depth:")
        self.param2_label.pack(pady=5)
        self.param2 = ttk.Combobox(root, values=[1, 2, 3])
        self.param2.pack(pady=5)
        self.param2.current(0)

        self.param2_label = ttk.Label(root, text="Alternating Questions:")
        self.param2_label.pack(pady=5)
        self.param2 = ttk.Combobox(root, values=[0, 1, 2])
        self.param2.pack(pady=5)
        self.param2.current(0)

        # Create a button to generate the graph
        self.generate_button = ttk.Button(root, text="Start", command=start_research_run())
        self.generate_button.pack(pady=20)

        # Create a placeholder for the graph
        self.canvas = None

    def generate_graph(self):
        # Generate some example data based on the selected parameters
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 15, 25, 30] if self.param1.get() == "Option 1" else [30, 25, 20, 15, 10]

        # Clear the previous graph if it exists
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        # Create a new figure and plot the data
        fig, ax = plt.subplots()
        ax.plot(x, y, label=f"{self.param1.get()} vs {self.param2.get()}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Performance Graph")
        ax.legend()

        # Display the graph in the GUI
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = Gui(root)
    root.mainloop()
