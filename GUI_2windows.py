import os
import tkinter as tk
from tkinter import filedialog
import anomalous
import numpy as np

padx = 10 # spacing between cells of the grid in x
pady = 10 # spacing between cells of the grid in y

def open_analysis_window():
    # Get user inputs from the first window
    path = path_entry.get()
    length = int(length_entry.get())
    analysis_type = analysis_type_var.get()

    # Close the first window
    root.destroy()

    # Create the analysis window
    analysis_window = tk.Tk()
    analysis_window.title("Anomalous Analysis - {}".format(analysis_type))

    if analysis_type == 'Fitting single tracks in Brownian motion':
        create_brownian_window(analysis_window, path, length)
    elif analysis_type == 'Fitting single tracks in confined motion':
        create_confined_window(analysis_window, path, length)

def create_brownian_window(window, path, length):

    # Initial LocErr Input
    locerr_label = tk.Label(window, text="Initial Localization error:")
    locerr_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    locerr_entry = tk.Entry(window, width=10)
    locerr_entry.grid(row=0, column=1)
    locerr_entry.insert(tk.END, "0.02")

    # Initial d Input
    d_label = tk.Label(window, text="Initial diffusion length:")
    d_label.grid(row=1, column=0, sticky="e", padx = padx, pady = pady)
    d_entry = tk.Entry(window, width=10)
    d_entry.grid(row=1, column=1)
    d_entry.insert(tk.END, "0.1")
    
    # Fixed LocErr Input
    fixed_locerr_label = tk.Label(window, text="Fixed LocErr:")
    fixed_locerr_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    fixed_locerr_var = tk.BooleanVar(value=True)
    fixed_locerr_check = tk.Checkbutton(window, variable=fixed_locerr_var)
    fixed_locerr_check.grid(row=2, column=1)

    # Number of Epochs Input
    epochs_label = tk.Label(window, text="Number of Epochs:")
    epochs_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    epochs_entry = tk.Entry(window, width=10)
    epochs_entry.grid(row=3, column=1)
    epochs_entry.insert(tk.END, "400")

    # Savepath Input
    savepath_label = tk.Label(window, text="Save Path:")
    savepath_label.grid(row=4, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = tk.Entry(window, width=50)
    savepath_entry.grid(row=4, column=1)
    savepath_entry.insert(tk.END, r'C:\Users\franc\OneDrive\Bureau\Anomalous\saved_results_brownian.csv')

    # Run Button
    run_button = tk.Button(window, text="Run Analysis", command=lambda: run_brownian_analysis(path, length, fixed_locerr_var.get(),
                                                                                              float(locerr_entry.get()), float(d_entry.get()),
                                                                                              int(epochs_entry.get()), savepath_entry.get()))
    run_button.grid(row=5, column=0, columnspan=2)

def create_confined_window(window, path, length):

    # Initial LocErr Input
    locerr_label = tk.Label(window, text="Initial Localization error:")
    locerr_label.grid(row=0, column=0, sticky = 'e', padx = padx, pady = pady)
    locerr_entry = tk.Entry(window, width=10)
    locerr_entry.grid(row=0, column=1)
    locerr_entry.insert(tk.END, "0.02")

    # Initial d Input
    d_label = tk.Label(window, text="Initial diffusion length of the particle:")
    d_label.grid(row=1, column=0, sticky = 'e', padx = padx, pady = pady)
    d_entry = tk.Entry(window, width=10)
    d_entry.grid(row=1, column=1)
    d_entry.insert(tk.END, "0.1")

    # Initial l Input
    l_label = tk.Label(window, text="Initial confinement factor:")
    l_label.grid(row=2, column=0, sticky = 'e', padx = padx, pady = pady)
    l_entry = tk.Entry(window, width=10)
    l_entry.grid(row=2, column=1)
    l_entry.insert(tk.END, "0.01")

    # Initial q Input
    q_label = tk.Label(window, text="Initial diffusion length of the potential well:")
    q_label.grid(row=3, column=0, sticky = 'e', padx = padx, pady = pady)
    q_entry = tk.Entry(window, width=10)
    q_entry.grid(row=3, column=1)
    q_entry.insert(tk.END, "0.01")
    
    # Fixed LocErr Input
    fixed_locerr_label = tk.Label(window, text="Fixed LocErr:")
    fixed_locerr_label.grid(row=4, column=0, sticky = 'e', padx = padx, pady = pady)
    fixed_locerr_var = tk.BooleanVar(value=True)
    fixed_locerr_check = tk.Checkbutton(window, variable=fixed_locerr_var)
    fixed_locerr_check.grid(row=4, column=1)

    # Number of Epochs Input
    epochs_label = tk.Label(window, text="Number of Epochs:")
    epochs_label.grid(row=5, column=0, sticky = 'e', padx = padx, pady = pady)
    epochs_entry = tk.Entry(window, width=10)
    epochs_entry.grid(row=5, column=1)
    epochs_entry.insert(tk.END, "400")

    # Savepath Input
    savepath_label = tk.Label(window, text="Save path:")
    savepath_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = tk.Entry(window, width=50)
    savepath_entry.grid(row=6, column=1)
    savepath_entry.insert(tk.END, r'C:\Users\franc\OneDrive\Bureau\Anomalous\saved_results_confinement.csv')

    # Run Button
    run_button = tk.Button(window, text="Run Analysis", command=lambda: run_confined_analysis(path, length, fixed_locerr_var.get(),
                                                                                              float(locerr_entry.get()), float(d_entry.get()),
                                                                                              float(q_entry.get()), float(l_entry.get()),
                                                                                              int(epochs_entry.get()), savepath_entry.get()))
    run_button.grid(row=7, column=0, columnspan=2)

def run_brownian_analysis(path, length, fixed_locerr, locerr, d, nb_epochs, savepath):
    # Run the Brownian motion analysis
    tracks, _, _ = anomalous.read_table(path, lengths=np.array([length]), dist_th=np.inf,
                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                         colnames=['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                         remove_no_disp=True)
    tracks = tracks[str(length)]
    pd_params = anomalous.Brownian_fit(tracks, verbose=1, Fixed_LocErr=fixed_locerr,
                                       Initial_params={'LocErr': locerr, 'd': d}, nb_epochs=nb_epochs)
    pd_params.to_csv(savepath)
    print("Brownian motion analysis completed and results saved to %s."%savepath)

def run_confined_analysis(path, length, fixed_locerr, locerr, d, q, l, nb_epochs, savepath):
    # Run the confined motion analysis
    tracks, _, _ = anomalous.read_table(path, lengths=np.array([length]), dist_th=np.inf,
                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                         colnames=['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                         remove_no_disp=True)
    tracks = tracks[str(length)]
    pd_params = anomalous.Confined_fit(tracks, verbose=1, Fixed_LocErr=fixed_locerr,
                                       Initial_params={'LocErr': locerr, 'd': d, 'q': q, 'l': l}, nb_epochs=nb_epochs)
    pd_params.to_csv(savepath)
    print("Confined motion analysis completed and results saved to %s."%savepath)

# Create the first window
root = tk.Tk()
root.title("Anomalous Analysis Setup")

# Path Input
path_label = tk.Label(root, text="Path:")
path_label.grid(row=0, column=0)
path_entry = tk.Entry(root, width=50)
path_entry.grid(row=0, column=1)
path_button = tk.Button(root, text="Browse", command=lambda: path_entry.insert(tk.END, filedialog.askopenfilename()))
path_button.grid(row=0, column=2)

# Length Input
length_label = tk.Label(root, text="Length:")
length_label.grid(row=1, column=0)
length_entry = tk.Entry(root, width=10)
length_entry.grid(row=1, column=1)
length_entry.insert(tk.END, "99")

# Analysis Type Input
analysis_type_label = tk.Label(root, text="Analysis Type:")
analysis_type_label.grid(row=2, column=0)
analysis_type_var = tk.StringVar(root)
analysis_type_var.set("Fitting single tracks in Brownian motion")
analysis_type_dropdown = tk.OptionMenu(root, analysis_type_var,
                                       "Fitting single tracks in Brownian motion",
                                       "Fitting single tracks in confined motion")
analysis_type_dropdown.grid(row=2, column=1)

# Next Button
next_button = tk.Button(root, text="Next", command=open_analysis_window)
next_button.grid(row=3, column=0, columnspan=3)

root.mainloop()




