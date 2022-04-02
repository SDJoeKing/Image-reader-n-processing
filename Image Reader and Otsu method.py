#%%
 
from nturl2path import pathname2url
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from skimage.filters import threshold_otsu
import os
# matplotlib.use('TkAgg')

# %%
def read(filepath, skiprows = 5):
    array = []
    with open(filepath) as f:
        lines = f.readlines()
        rows = skiprows
        for l in lines:
            if rows != 0:
                rows -= 1
                continue
            l = l.split(' ')
            l = [i for i in l if i!='']
            l[-1] = l[-1].strip('\n')
            array.append(l)
    return np.array(array, dtype=np.float)

def draw_figure(already_packed, canvas, figure):
    """ Draw a matplotlib figure onto a Tk canvas
    Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
    """
    if canvas in already_packed:
        figure_canvas_agg =  already_packed[canvas]
    else:
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        already_packed[canvas] = figure_canvas_agg
    figure_canvas_agg.draw()
    return figure_canvas_agg

layout = [[sg.Text('Select Filepath')],
          [sg.Input(), sg.FileBrowse()],
          [sg.Text("Otsu's threshold:"), sg.Input(0,key='-O-', size=(5,1)), sg.Button('Apply threshold', key='-A-', disabled = True) ],
          [sg.Canvas(key='-CANVAS-')],
          [sg.OK('Plot', key = '-PLOT-'), sg.Cancel(key = '-C-'), sg.Button('Save',key = '-S-')]]

window = sg.Window('Automatic Image Reader and Otsu method', layout, finalize=True, element_justification='center', font='Helvetica 18')
fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
already_packed = {}
while True:
    event, values = window.read()

    if event in (None, '-C-'):
        break
   
    if event == '-PLOT-':
        # file = pd.read_csv(values['Browse'], delimiter = ' ', skiprows = 5, header = None).dropna(axis = 1)
        window['-A-'].update(disabled = False)
        file = read(values['Browse'], skiprows = 5)
        file = np.apply_along_axis(lambda x:x-273.15, 1, file) # Kelvin to Celsius
        file = np.array(file)
       
        thresh = threshold_otsu(file)
        window['-O-'].update(thresh)    
       
       
        ax.imshow(file)
        fig_canvas_agg = draw_figure(already_packed,window['-CANVAS-'].TKCanvas, fig)
   
    if event == '-A-':
        ax.imshow(file > float(values['-O-']))
        fig_canvas_agg = draw_figure(already_packed,window['-CANVAS-'].TKCanvas, fig)
       
    if event == '-S-':
        txt = sg.popup_get_text('File Name?')
        if txt:
            fig.savefig(os.getcwd()+'\\'+txt,dpi=300)
 
           
         
window.close()
#%%