import numpy as np
# from scipy.interpolate import RegularGridInterpolator as rgi
# from . import trigutils
# from . import utils

# class BispectrumInterpolator:
#     def __init__(self, bispectrum, lmin, lmax, nlbin, npsibin, nmubin, psipad=0, mupad=0, domain=3):
#         """
#         bispectrum (callable): bispectrum function
#         lmin (float): minimum l
#         lmax (float): maximum l
#         nlbin (int): number of l bins
#         npsibin (int): number of psi bins
#         nmubin (int): number of mu bins
#         psipad (float): padding for psi bins
#         mupad (float): padding for mu bins
#         domain (int): domain id of bispectrum function
#         """

#         # callable bispectrum function
#         self.bispectrum = bispectrum
#         # bin ranges
#         self.lmin   = lmin
#         self.lmax   = lmax
#         self.psipad = psipad
#         self.mupad  = mupad
#         # bin size
#         self.nlbin  = nlbin
#         self.npsibin= npsibin
#         self.nmubin = nmubin
#         # domain id
#         self.domain = domain

#     def init_spline(self, method='linear'):
#         """
#         Note that the methods other than 'linear' makes __call__ method significantly slower.
#         """
#         if self.domain == 3:
#             args, out = self.get_table_in_domain3()
#         else:
#             raise NotImplementedError
#         # spline
#         self.interp = rgi(args, np.log(out), method=method)

#     def get_table_in_domain3(self):
#         self.l = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nlbin)
#         self.a = np.linspace(0, 1, self.npsibin)
#         self.b = np.linspace(0, 1, self.nmubin)

#         # the parametrization of psi and mu by latent variable a and b
#         # is little a bit tricky. The following is parametrization makes
#         # the bispectrum smooth enough to interpolate as a function of (l, a, b).
#         # Note that no physical meaning is assigned to this parametrization.
#         mumin = (-1+self.mupad)
#         L, A, B = np.meshgrid(self.l, self.a, self.b, indexing='ij')
#         Psi = self.psipad + A**3*(np.pi/4.0 - self.psipad)
#         Mu  = mumin       + B   *(0.5*np.tan(Psi) - mumin)

#         # side lengths of triangle in Fourier space
#         L1 = L*np.cos(Psi)
#         L2 = L*np.sin(Psi)
#         L3 = L*np.sqrt(1-np.sin(2*Psi)*Mu)

#         # Create table
#         out = L**2*self.bispectrum(L1, L2, L3)

#         # 1d args
#         args = (np.log(self.l), self.a, self.b)

#         return args, out

#     def get_latent_arguments(self, l1, l2, l3):
#         if self.domain == 3:
#             return self.get_latent_arguments_in_domain3(l1, l2, l3)
#         else:
#             raise NotImplementedError

#     def get_latent_arguments_in_domain3(self, l1, l2, l3):
#         # order
#         l2, l1, l3 = np.sort([l1, l2, l3], axis=0)

#         # l, psi, mu
#         l, psi, mu = trigutils.x1x2x3_to_xpsimu(l1, l2, l3)

#         # convert latent space argument
#         mumin = (-1+self.mupad)
#         a = ((psi - self.psipad)/(np.pi/4.0 - self.psipad))**(1.0/3.0)
#         b = (mu - mumin)/(0.5*np.tan(psi) - mumin)
        
#         # force small excess/deficit of a, b from 0/1 
#         # to be zero due to finite machine precision
#         a[a<0] = 0
#         a[a>1] = 1
#         b[b<0] = 0
#         b[b>1] = 1

#         return l, a, b

#     def __call__(self, l1, l2, l3, replace_close=True):
#         # get latent arguments
#         l, a, b = self.get_latent_arguments(l1, l2, l3)

#         # check boundary
#         if replace_close:
#             l = utils.replace_close(l, self.l.min(), self.l.max())
#             a = utils.replace_close(a, self.a.min(), self.a.max())
#             b = utils.replace_close(b, self.b.min(), self.b.max())

#         # interpolate
#         bk = self.interp((np.log(l), a, b))
#         bk = np.exp(bk) / l**2

#         return bk

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
