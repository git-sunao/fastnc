#!/usr/bin/env python
'''
Description:
This is the interpolation module of fastnc.

Author     : Sunao Sugiyama 
Last edit  : 2024/01/21 21:16:02
'''
import numpy as np

class SemiDiagonalInterpolator:
    def __init__(self, x, y, f):
        # data
        self.xmin = x.min()
        self.xmax = x.max()
        self.n    = x.size
        self.dx   = x[1] - x[0]
        self.ymin = y.min()
        self.ymax = y.max()
        self.m    = y.size
        self.dy   = y[1] - y[0]
        
        # standardize the grid
        # now x and y are in [0, n] and [0, m]
        self.x = self.standardize_x(x)
        self.y = self.standardize_y(y)

        # function
        self.f = f

    def standardize_x(self, x):
        return (x - self.xmin)/self.dx

    def standardize_y(self, y):
        return (y - self.ymin)/self.dy

    def __call__(self, x, y):
        x = self.standardize_x(x)
        y = self.standardize_y(y)

        a = y[0] - x[0]
        assert np.all(np.isclose(a, y-x)), 'x and y must be on a diagonal line'

        a0= np.floor(a).astype(int)
        a1= a0 + 1

        if a<=0:
            f0= np.interp(x, self.x[abs(a0):], np.diag(self.f[abs(a0):,None:]))
            f1= np.interp(x, self.x[abs(a1):], np.diag(self.f[abs(a1):,None:]))
            f = f0*(a1-a) + f1*(a-a0)
        else:
            f0= np.interp(x, self.x[:self.n-abs(a0)], np.diag(self.f[None:, abs(a0):]))
            f1= np.interp(x, self.x[:self.n-abs(a1)], np.diag(self.f[None:, abs(a1):]))
            f = f0*(a1-a) + f1*(a-a0)

        return f
