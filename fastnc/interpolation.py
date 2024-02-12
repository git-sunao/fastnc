#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/02/06 21:31:07

Description:
This is the interpolation module of fastnc.
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

    def __get_semidiag(self, a):
        if a<=0:
            x = self.x[abs(a):]
            y = self.y[:self.m-abs(a)]
            f = self.f[abs(a):,None:]
        else:
            x = self.x[:self.n-abs(a)]
            y = self.y[abs(a):]
            f = self.f[None:, abs(a):]
        return x, y, f

    def __get_semidiag_interpolate_from_x(self, a, x):
        sx, sy, sf = self.__get_semidiag(a)
        return np.interp(x, sx, np.diag(sf))

    def __get_semidiag_interpolate_from_y(self, a, x):
        sx, sy, sf = self.__get_semidiag(a)
        return np.interp(x, sx, np.diag(sf))
        
    def __call__(self, x, y, method='linear'):
        x = self.standardize_x(x)
        y = self.standardize_y(y)

        # io check
        a = y[0] - x[0]
        assert np.all(np.isclose(a, y-x)), 'x and y must be on a diagonal line'
        assert method in ['linear', 'cubic'], 'method must be either linear or cubic'

        a0 = np.floor(a).astype(int)
        a1 = a0 + 1
        a2 = a0 + 2
        a_1= a0 - 1

        # get necessary diagonals
        f0 = self.__get_semidiag_interpolate_from_x(a0, x)
        f1 = self.__get_semidiag_interpolate_from_x(a1, x)
        if method == 'cubic':
            f2 = self.__get_semidiag_interpolate_from_x(a2, x)
            f_1= self.__get_semidiag_interpolate_from_x(a_1, x)

        # interpolate
        if method == 'linear':
            f = f0*(a1-a) + f1*(a-a0)
        elif method == 'cubic':
            f = f0*(a1-a)*(a2-a)/(a1-a0)/(a2-a0) + f1*(a-a0)*(a2-a)/(a1-a0)/(a2-a1) + f_1*(a-a0)*(a1-a)/(a_1-a0)/(a1-a_1) + f2*(a-a_1)*(a1-a)/(a2-a_1)/(a1-a_1)
        
        return f
