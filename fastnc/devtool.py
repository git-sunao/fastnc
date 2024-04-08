#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/25 21:50:38

Description:
This contains some useful functions for development:
plotting, triangle plotter, stopwatch, etc.
'''
import matplotlib.pyplot as plt
import numpy as np

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
