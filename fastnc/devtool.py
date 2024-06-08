#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/06/07 11:38:43

Description:
This contains some useful functions for development:
plotting, triangle plotter, stopwatch, etc.
'''
import matplotlib.pyplot as plt
import numpy as np
import time

class Timer:
    def __init__(self):
        self.lap_time = np.array([])
        self.lap_name = np.array([], dtype=str)
        self('start')
    def __call__(self, name=''):
        self.lap_time = np.append(self.lap_time, time.time())
        self.lap_name = np.append(self.lap_name, name)
    def get_name_runtime(self, group=True):
        names = self.lap_name[1:]
        times = np.diff(self.lap_time)
        if group:
            ind = np.unique(names, return_index=True)[1]
            # time
            times2 = []
            for i in sorted(ind):
                name = names[i]
                w = names == name
                times2.append(np.sum(times[w]))
            times = np.array(times2)
            # name
            names = np.array([names[i] for i in sorted(ind)])
        return names, times
    def summarize(self, group=True):
        names, times = self.get_name_runtime(group=group)
        ljust = max([len(name) for name in names])
        text = ''
        for name, dt in zip(names, times):
            text += f'{name.ljust(ljust)}: {dt:.3f} s\n'
        text += f'Total'.ljust(ljust)+f': {self.lap_time[-1]-self.lap_time[0]:.3f} s\n'
        return text
    def latex(self, wrap=True, group=True):
        text = ''
        hline = '\\hline'
        hline2= '\\hline\\hline'
        names, times = self.get_name_runtime(group=group)
        ljust = max([len(name) for name in names])
        for name, dt in zip(names, times):
            text += f'{name.ljust(ljust)} & {dt:.3f} s \\\\' + '\n'
        text += hline + '\n'
        text += f'Total'.ljust(ljust)+f' & {self.lap_time[-1]-self.lap_time[0]:.3f} s \\\\' + '\n'
        if wrap:
            text = '\\begin{tabular}{|l|c|}\n'+hline2+'\n'+ text +hline2+'\n'+ '\\end{tabular}'
        return text

# triangle plotter
def plot_triangle(ax, x1, x2, x3, loc='lower left', bbox_to_anchor=(0.75, 0.05), scale=0.2, sort=True, edgecolor='k', **kwargs):
    if sort:
        x3, x2, x1 = np.sort([x1, x2, x3])

    # ax aspect ratio
    w, h = ax.get_figure().get_size_inches()
    aspect = h/w

    # vertex
    X3 = (0., 0.0)
    X2 = (x1, 0.0)
    h  = (x1**2+x2**2-x3**2)/2/x1
    y  = np.sqrt(x2**2-h**2)
    X1 = (h, y/aspect)

    # vtx
    vtx = np.array([X1, X2, X3])

    # scale vtx
    factor = scale/x1
    vtx *= factor

    # Shift horizontally
    horizon = loc.split(' ')[1]
    if horizon == 'left':
        vtx[:,0] += bbox_to_anchor[0]
    elif horizon == 'right':
        vtx[:,0] -= bbox_to_anchor[0]
    else:
        raise ValueError('Invalid loc: {}'.format(loc))
    
    # Shift vertically
    vertical = loc.split(' ')[0]
    if vertical == 'lower':
        vtx[:,1] += bbox_to_anchor[1]
    elif vertical == 'upper':
        vtx[:,1] -= bbox_to_anchor[1]
    else:
        raise ValueError('Invalid loc: {}'.format(loc))

    # Plot triangle
    triangle = plt.Polygon(vtx, edgecolor=edgecolor, facecolor='none', transform=ax.transAxes, **kwargs)
    ax.add_patch(triangle)
