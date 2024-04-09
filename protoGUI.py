


import os
import tkinter as tk
from tkinter import filedialog
import anomalous
import numpy as np

def run_anomalous_analysis():
    # Get user inputs from GUI
    path = path_entry.get()
    savepath = savepath_entry.get()
    length = int(length_entry.get())
    Fixed_LocErr = fixed_locerr_var.get()
    Initial_params = {'LocErr': float(locerr_entry.get()), 'd': float(d_entry.get())}
    nb_epochs = int(epochs_entry.get())

    # Run the anomalous analysis
    tracks, _, _ = anomalous.read_table(path, lengths=np.array([length]), dist_th=np.inf,
                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                         colnames=['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                         remove_no_disp=True)
    tracks = tracks[str(length)]
    pd_params = anomalous.Brownian_fit(tracks, verbose=1, Fixed_LocErr=Fixed_LocErr,
                                       Initial_params=Initial_params, nb_epochs=nb_epochs)
    pd_params.to_csv(savepath)
    status_label.config(text="Anomalous analysis completed and results saved.")

# Create GUI
root = tk.Tk()
root.title("Anomalous Analysis")

# Path Input
path_label = tk.Label(root, text="Path:")
path_label.grid(row=0, column=0)
path_entry = tk.Entry(root, width=50)
path_entry.grid(row=0, column=1)
path_button = tk.Button(root, text="Browse", command=lambda: path_entry.insert(tk.END, filedialog.askopenfilename()))
path_button.grid(row=0, column=2)

# Savepath Input
savepath_label = tk.Label(root, text="Save Path:")
savepath_label.grid(row=1, column=0)
savepath_entry = tk.Entry(root, width=50)
savepath_entry.grid(row=1, column=1)
savepath_button = tk.Button(root, text="Browse", command=lambda: savepath_entry.insert(tk.END, filedialog.asksaveasfilename()))
savepath_button.grid(row=1, column=2)

# Length Input
length_label = tk.Label(root, text="Length:")
length_label.grid(row=2, column=0)
length_entry = tk.Entry(root, width=10)
length_entry.grid(row=2, column=1)
length_entry.insert(tk.END, "99")

# Fixed LocErr Input
fixed_locerr_var = tk.BooleanVar()
fixed_locerr_var.set(True)
fixed_locerr_check = tk.Checkbutton(root, text="Fixed LocErr", variable=fixed_locerr_var)
fixed_locerr_check.grid(row=3, column=0)

# Initial LocErr Input
locerr_label = tk.Label(root, text="Initial LocErr:")
locerr_label.grid(row=4, column=0)
locerr_entry = tk.Entry(root, width=10)
locerr_entry.grid(row=4, column=1)
locerr_entry.insert(tk.END, "0.02")

# Initial d Input
d_label = tk.Label(root, text="Initial d:")
d_label.grid(row=5, column=0)
d_entry = tk.Entry(root, width=10)
d_entry.grid(row=5, column=1)
d_entry.insert(tk.END, "0.1")

# Number of Epochs Input
epochs_label = tk.Label(root, text="Number of Epochs:")
epochs_label.grid(row=6, column=0)
epochs_entry = tk.Entry(root, width=10)
epochs_entry.grid(row=6, column=1)
epochs_entry.insert(tk.END, "400")

# Run Button
run_button = tk.Button(root, text="Run Anomalous Analysis", command=run_anomalous_analysis)
run_button.grid(row=7, column=0, columnspan=3)

# Status Label
status_label = tk.Label(root, text="")
status_label.grid(row=8, column=0, columnspan=3)

root.mainloop()




