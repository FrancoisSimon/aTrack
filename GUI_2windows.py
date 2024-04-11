import os
import tkinter as tk
from tkinter import filedialog
import anomalous
import numpy as np

padx = 10 # spacing between cells of the grid in x
pady = 10 # spacing between cells of the grid in y
previous_window = None 

def open_analysis_window():
    global previous_window
    path = path_entry.get()
    print(os.path.normpath(path))
    savepath = os.path.normpath(path).rsplit(os.sep, 1)[0]
    (os.sep, 1)[0]
    length = int(length_entry.get())
    analysis_type = analysis_type_var.get()

    root.withdraw()
    previous_window = root 

    analysis_window = tk.Tk()
    analysis_window.title("Anomalous Analysis - {}".format(analysis_type))

    if analysis_type == 'Fitting single tracks in Brownian motion':
        create_brownian_window(analysis_window, path, savepath, length)
    elif analysis_type == 'Fitting single tracks in confined motion':
        create_confined_window(analysis_window, path, savepath, length)
    elif analysis_type == 'Fitting single tracks in directed motion':
        create_directed_window(analysis_window, path, savepath, length)
    elif analysis_type == 'Fitting model with multiple states':
        create_multi_window(analysis_window, path, savepath, length)

def go_to_previous_window(window):
    window.destroy()
    if previous_window:
        previous_window.deiconify()

def create_brownian_window(window, path, savepath, length):

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
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_results_brownian.csv'))

    # Run Button
    run_button = tk.Button(window, text="Run Analysis", command=lambda: run_brownian_analysis(path, length, fixed_locerr_var.get(),
                                                                                              float(locerr_entry.get()), float(d_entry.get()),
                                                                                              int(epochs_entry.get()), savepath_entry.get()))
    run_button.grid(row=5, column=1, columnspan=2)

    # Previous Button
    previous_button = tk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=5, column=0, columnspan=2)

def create_confined_window(window, path, savepath, length):

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
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_results_confined.csv'))

    # Run Button
    run_button = tk.Button(window, text="Run Analysis", command=lambda: run_confined_analysis(path, length, fixed_locerr_var.get(),
                                                                                              float(locerr_entry.get()), float(d_entry.get()),
                                                                                              float(q_entry.get()), float(l_entry.get()),
                                                                                              int(epochs_entry.get()), savepath_entry.get()))
    run_button.grid(row=7, column=1, columnspan=2)

    # Previous Button
    previous_button = tk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=7, column=0, columnspan=2)

def create_directed_window(window, path, savepath, length):

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
    d_entry.insert(tk.END, "0.01")

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
    savepath_entry.insert(tk.END, os.path.join(savepath, 'saved_results_directed.csv'))

    # Run Button
    run_button = tk.Button(window, text="Run Analysis", command=lambda: run_directed_analysis(path, length, fixed_locerr_var.get(),
                                                                                              float(locerr_entry.get()), float(d_entry.get()),
                                                                                              float(q_entry.get()), float(l_entry.get()),
                                                                                              int(epochs_entry.get()), savepath_entry.get()))
    run_button.grid(row=7, column=1, columnspan=2)

    # Previous Button
    previous_button = tk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=7, column=0, columnspan=2)

def create_multi_window(window, path, savepath, length):

    # Fixed LocErr Input
    fixed_locerr_label = tk.Label(window, text="Fixed LocErr:")
    fixed_locerr_label.grid(row=0, column=1, sticky = 'e', padx = padx, pady = pady)
    fixed_locerr_var = tk.BooleanVar(value=True)
    fixed_locerr_check = tk.Checkbutton(window, variable=fixed_locerr_var)
    fixed_locerr_check.grid(row=0, column=2)

    # Number of Epochs Input
    epochs_label = tk.Label(window, text="Number of Epochs:")
    epochs_label.grid(row=1, column=1, sticky = 'e', padx = padx, pady = pady)
    epochs_entry = tk.Entry(window, width=10)
    epochs_entry.grid(row=1, column=2)
    epochs_entry.insert(tk.END, "1000")

    # Min Number of States Input
    min_nb_states_label = tk.Label(window, text="Minimal Number of States:")
    min_nb_states_label.grid(row=2, column=1, sticky = 'e', padx = padx, pady = pady)
    min_nb_states_entry = tk.Entry(window, width=10)
    min_nb_states_entry.grid(row=2, column=2)
    min_nb_states_entry.insert(tk.END, "3")

    # Max Number of States Input
    max_nb_states_label = tk.Label(window, text="Maximal Number of States:")
    max_nb_states_label.grid(row=3, column=1, sticky = 'e', padx = padx, pady = pady)
    max_nb_states_entry = tk.Entry(window, width=10)
    max_nb_states_entry.grid(row=3, column=2)
    max_nb_states_entry.insert(tk.END, "15")

    # Batch size Input
    batch_size_label = tk.Label(window, text="Number of tracks considered per batch:")
    batch_size_label.grid(row=4, column=1, sticky = 'e', padx = padx, pady = pady)
    batch_size_entry = tk.Entry(window, width=10)
    batch_size_entry.grid(row=4, column=2)
    batch_size_entry.insert(tk.END, "2048")

    # Savepath Input
    savepath_label = tk.Label(window, text="Save path:")
    savepath_label.grid(row=5, column=1, sticky = 'e', padx = padx, pady = pady)
    savepath_entry = tk.Entry(window, width=50)
    savepath_entry.grid(row=5, column=2)
    savepath_entry.insert(tk.END, os.path.join(os.getcwd(), 'saved_results_multi.csv'))

    # Initial Confined LocErr Input
    confined_locerr_label = tk.Label(window, text="Initial Confined Localization error:")
    confined_locerr_label.grid(row=6, column=0, sticky = 'e', padx = padx, pady = pady)
    confined_locerr_entry = tk.Entry(window, width=10)
    confined_locerr_entry.grid(row=6, column=1)
    confined_locerr_entry.insert(tk.END, "0.02")

    # Initial Confined d Input
    confined_d_label = tk.Label(window, text="Initial diffusion length of the particle for confined fitting:")
    confined_d_label.grid(row=7, column=0, sticky = 'e', padx = padx, pady = pady)
    confined_d_entry = tk.Entry(window, width=10)
    confined_d_entry.grid(row=7, column=1)
    confined_d_entry.insert(tk.END, "0.1")

    # Initial Confined l Input
    confined_l_label = tk.Label(window, text="Initial confinement factor for confined fitting:")
    confined_l_label.grid(row=8, column=0, sticky = 'e', padx = padx, pady = pady)
    confined_l_entry = tk.Entry(window, width=10)
    confined_l_entry.grid(row=8, column=1)
    confined_l_entry.insert(tk.END, "0.01")

    # Initial Confined q Input
    confined_q_label = tk.Label(window, text="Initial diffusion length of the potential well for confined fitting:")
    confined_q_label.grid(row=9, column=0, sticky = 'e', padx = padx, pady = pady)
    confined_q_entry = tk.Entry(window, width=10)
    confined_q_entry.grid(row=9, column=1)
    confined_q_entry.insert(tk.END, "0.01")

    # Initial Directed LocErr Input
    directed_locerr_label = tk.Label(window, text="Initial Directed Localization error:")
    directed_locerr_label.grid(row=6, column=2, sticky = 'e', padx = padx, pady = pady)
    directed_locerr_entry = tk.Entry(window, width=10)
    directed_locerr_entry.grid(row=6, column=3)
    directed_locerr_entry.insert(tk.END, "0.02")

    # Initial Directed d Input
    directed_d_label = tk.Label(window, text="Initial diffusion length of the particle for directed fitting:")
    directed_d_label.grid(row=7, column=2, sticky = 'e', padx = padx, pady = pady)
    directed_d_entry = tk.Entry(window, width=10)
    directed_d_entry.grid(row=7, column=3)
    directed_d_entry.insert(tk.END, "0.1")

    # Initial Confined l Input
    directed_l_label = tk.Label(window, text="Initial confinement factor for directed fitting:")
    directed_l_label.grid(row=8, column=2, sticky = 'e', padx = padx, pady = pady)
    directed_l_entry = tk.Entry(window, width=10)
    directed_l_entry.grid(row=8, column=3)
    directed_l_entry.insert(tk.END, "0.01")

    # Initial Confined q Input
    directed_q_label = tk.Label(window, text="Initial diffusion length of the potential well for directed fitting:")
    directed_q_label.grid(row=9, column=2, sticky = 'e', padx = padx, pady = pady)
    directed_q_entry = tk.Entry(window, width=10)
    directed_q_entry.grid(row=9, column=3)
    directed_q_entry.insert(tk.END, "0.01")

    # Run Button
    run_button = tk.Button(window, text="Run Analysis", command=lambda: run_multi_analysis(path, length, fixed_locerr_var.get(), int(min_nb_states_entry.get()), int(max_nb_states_entry.get()),
                                                                                              float(confined_locerr_entry.get()), float(confined_d_entry.get()),
                                                                                              float(confined_q_entry.get()), float(confined_l_entry.get()),
                                                                                              float(directed_locerr_entry.get()), float(directed_d_entry.get()),
                                                                                              float(directed_q_entry.get()), float(directed_l_entry.get()),
                                                                                              int(epochs_entry.get()), int(batch_size_entry.get()), savepath_entry.get()))
    run_button.grid(row=10, column=1, columnspan=2)

    # Previous Button
    previous_button = tk.Button(window, text="Previous", command=lambda: go_to_previous_window(window))
    previous_button.grid(row=10, column=0, columnspan=2)

def run_brownian_analysis(path, length, fixed_locerr, locerr, d, nb_epochs, savepath):
    # Run the Brownian motion analysis
    tracks, _, _ = anomalous.read_table(path, lengths=np.array([length]), dist_th=np.inf,
                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                         colnames=['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                         remove_no_disp=True)
    tracks = tracks[str(length)]
    pd_params = anomalous.Brownian_fit(tracks, verbose=1, Fixed_LocErr=fixed_locerr,
                                       Initial_params={'LocErr': locerr, 'd': d}, nb_epochs=nb_epochs)
    print(savepath)
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

def run_directed_analysis(path, length, fixed_locerr, locerr, d, q, l, nb_epochs, savepath):
    # Run the directed motion analysis
    tracks, _, _ = anomalous.read_table(path, lengths=np.array([length]), dist_th=np.inf,
                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                         colnames=['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                         remove_no_disp=True)
    tracks = tracks[str(length)]
    pd_params = anomalous.Directed_fit(tracks, verbose=1, Fixed_LocErr=fixed_locerr,
                                       Initial_params={'LocErr': locerr, 'd': d, 'q': q, 'l': l}, nb_epochs=nb_epochs)
    pd_params.to_csv(savepath)
    print("Directed motion analysis completed and results saved to %s."%savepath)

def run_multi_analysis(path, length, fixed_locerr, min_nb_states, max_nb_states, confined_locerr, confined_d, confined_q, confined_l, directed_locerr, directed_d, directed_q, directed_l, nb_epochs, batch_size, savepath):
    # Run the multiple states analysis
    tracks, _, _ = anomalous.read_table(path, lengths=np.array([length]), dist_th=np.inf,
                                         frames_boundaries=[-np.inf, np.inf], fmt='csv',
                                         colnames=['POSITION_X', 'POSITION_Y', 'FRAME', 'TRACK_ID'],
                                         remove_no_disp=True)
    tracks = tracks[str(length)]
    likelihoods, all_pd_params = anomalous.multi_fit(tracks, verbose=1, Fixed_LocErr=fixed_locerr, min_nb_states=min_nb_states, max_nb_states=max_nb_states, nb_epochs=nb_epochs, batch_size=batch_size, 
                                    Initial_confined_params={'LocErr': confined_locerr, 'd': confined_d, 'q': confined_q, 'l': confined_l},
                                    Initial_directed_params={'LocErr': directed_locerr, 'd': directed_d, 'q': directed_q, 'l': directed_l}, 
                                    )
    
    for nb_states in all_pd_params.keys():
        pd_params = all_pd_params[nb_states]
        pd_params.to_csv(savepath[:-4] + '_' + nb_states + savepath[-4:])
        
    likelihoods.to_csv(savepath[:-4] + '_likelihood' + savepath[-4:])
    print("Multiple states model analysis completed and results saved to %s."%savepath)

# Create the first window
root = tk.Tk()
root.title("Anomalous Analysis Setup")

def browser():
    path_entry.delete(0,'end')
    path_entry.insert(tk.END, filedialog.askopenfilename(initialdir=os.path.expanduser('~'), title="Select File"))

# Path Input
path_label = tk.Label(root, text="Path:")
path_label.grid(row=0, column=0)
path_entry = tk.Entry(root, width=50)
path_entry.grid(row=0, column=1)
path_entry.insert(tk.END, os.getcwd())
#path_button = tk.Button(root, text="Browse", command=lambda: path_entry.insert(tk.END, filedialog.askopenfilename()))
#path_button = tk.Button(root, text="Browse", command=lambda: (path_entry.insert(tk.END, filedialog.askopenfilename(initialdir=os.path.expanduser('~'), title="Select File"))))
path_button = tk.Button(root, text="Browse", command=browser)
path_button.grid(row=0, column=2)
dir(path_entry)
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
                                       "Fitting single tracks in confined motion",
                                       "Fitting single tracks in directed motion",
                                       "Fitting model with multiple states")
analysis_type_dropdown.grid(row=2, column=1)

# Next Button
next_button = tk.Button(root, text="Next", command=open_analysis_window)
next_button.grid(row=3, column=0, columnspan=3)

root.mainloop()


import pandas as pd
a = pd.DataFrame(np.zeros((3,3)))

a.to_csv(r'C:/Users/franc/Downloads/aTrack-main/aTrack-main/example_tracks.csv\saved_results_brownian.csv')





