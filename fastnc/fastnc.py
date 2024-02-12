#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/02/08 20:40:46

Description:
This is the module of fastnc, which calculate the
natural components using 2d fftlog.
'''
import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm
from glob import glob
import pandas as pd
import os
# fastnc modules
from . import twobessel
from . import trigutils
from . import utils
from .interpolation import SemiDiagonalInterpolator as sdi

def sincos2angbar(psi, delta):
    cos2b = np.cos(delta) + np.sin(2*psi)
    sin2b = np.cos(2*psi) * np.sin(delta)
    norm  = np.sqrt(cos2b**2 + sin2b**2)
    return sin2b/norm, cos2b/norm

class GLMdatabase:
    def __init__(self, csv_filename):
        self.csv_filename = csv_filename
        self.df = self.load_database()

    def load_database(self):
        # Load the CSV file into memory if it exists, otherwise initialize an empty dataframe
        if os.path.exists(self.csv_filename):
            return pd.read_csv(self.csv_filename)
        else:
            return pd.DataFrame(columns=['Npsi', 'tol', 'filename'])

    def find_entry(self, Npsi, tol):
        # Check if entry with given Npsi and tol already exists
        existing_entry = self.df[(self.df['Npsi'] == Npsi) & (self.df['tol'] == tol)]

        if not existing_entry.empty:
            # If entry exists, return the filename associated with it
            return existing_entry['filename'].values[0]
        else:
            return None

    def find_or_create_entry(self, Npsi, tol):
        # Check if entry with given Npsi and tol already exists
        existing_entry = self.df[(self.df['Npsi'] == Npsi) & (self.df['tol'] == tol)]

        if not existing_entry.empty:
            # If entry exists, return the filename associated with it
            return existing_entry['filename'].values[0]
        else:
            # If entry does not exist, generate a new unique filename
            new_filename = self.generate_unique_filename()
            
            # Add a new row to the dataframe with the provided Npsi, tol, and generated filename
            new_row = pd.DataFrame({'Npsi': [Npsi], 'tol': [tol], 'filename': [new_filename]})
            self.df = pd.concat([self.df, new_row], ignore_index=True)

            # Write the updated dataframe to the CSV file
            self.write()

            # Return the newly generated filename
            return new_filename

    def generate_unique_filename(self):
        # Generate a new filename that does not already exist in the dataframe
        i = 1
        while True:
            new_filename = f"GLMdata-{i}.pkl"
            if new_filename not in self.df['filename'].values:
                return new_filename
            i += 1

    def write(self):
        # Write the current dataframe to the CSV file
        self.df.to_csv(self.csv_filename, index=False)

class GLMCalculator:
    """
    Calculate the GLM coefficients.
    .. math:
        G_LM(psi) = 2 \\int_0^{\\pi} dx P_L(\\cos(x)) \\cos[2\\bar\\beta(\\psi, x) + M x]

    Parameters:
    - Lmax (int): Maximum multipole moment.
    - Max (int): Maximum angular Fourier mode.
    - Npsi (int): Number of psi bins.

    Attributes:
    - Lmax (int): Maximum multipole moment.
    - Max (int): Maximum angular Fourier mode.
    - psi (array): Psi bins.
    """

    cachedir = os.path.join(os.path.dirname(__file__), 'cache')
    databasename = os.path.join(cachedir, 'GLMdatabase.csv')

    def __init__(self, Lmax, Mmax, Npsi=200, tol=1e-4, verbose=False, cache=True):
        self.Lmax = Lmax
        self.Mmax = Mmax
        self.psi  = np.linspace(0, np.pi/2, Npsi)
        self.tol  = tol

        # general setup
        self.verbose = verbose

        # initialize GLMdata
        self.GLMdata = dict()

        # load GLMdata from cache if exists
        self.load_cache()

        # compute/update GLM 
        self.compute_GLM(self.tol, verbose=self.verbose)

        # save cache
        if cache and not os.path.exists(self.databasename):
            os.makedirs(self.cachedir, exist_ok=True)
        if cache:
            self.save_cache()

    def integrand(self, x, psi, L, M):
        x, psi = np.meshgrid(x, psi, indexing='ij')
        sin2bb, cos2bb = sincos2angbar(psi, x)
        out = 4*np.pi*eval_legendre(L, np.cos(x)) * (cos2bb*np.cos(M*x) - sin2bb*np.sin(M*x))
        return out
        
    def compute_GLM(self, tol, correct_bias=True, verbose=False):
        from .integration import aint
        # prepare todo list
        todo = []
        for L in range(self.Lmax+1):
            for M in range(self.Mmax+1):
                todo.append([L, M])

        # compute GLM
        pbar = tqdm(todo, desc='[GLM]', disable=not verbose)
        for L, M in pbar:
            pbar.set_postfix({'L':L, 'M':M})
            if (L,M) in self.GLMdata:
                continue
            args = {'L':L, 'M':M, 'psi':self.psi}
            o, c = aint(self.integrand, 0, np.pi, 2, tol=tol, **args)
            self.GLMdata[(L, M)] = o

        # Correction
        # G_LM(psi) is exactly zero for L<M and psi<=pi/4. 
        # However, the numerical integration may give a non-zero value.
        # From my experience, the same amount of error is also present in
        # G_ML if G_LM is biased. Hence we subtract the error in G_LM(psi<=pi/4)
        # from G_LM(psi) and G_ML(psi).
        if correct_bias:
            for (L,M), data in self.GLMdata.items():
                if L>=M:
                    continue
                # estimate the bias in G_LM(psi<=np.pi/4)
                bias = np.mean(data[self.psi<=np.pi/4])
                # subtract the bias
                self.GLMdata[(L,M)] -= bias
                if (M,L) in self.GLMdata:
                    self.GLMdata[(M,L)] -= bias
        
    def save_cache(self):
        # load database
        database = GLMdatabase(self.databasename)

        # find or create entry in the database
        filename = database.find_or_create_entry(len(self.psi), self.tol)

        # save
        utils.save_pickle(os.path.join(self.cachedir, filename), self.GLMdata)

    def load_cache(self):
        # load database
        database = GLMdatabase(self.databasename)

        # filename
        filename = database.find_entry(len(self.psi), self.tol)

        # load
        if filename is not None:
            print('Load GLMdata from cache: {}'.format(filename)) if self.verbose else None
            self.GLMdata = utils.load_pickle(os.path.join(self.cachedir, filename))

    def __call__(self, L, M, psi):
        """
        Compute GLM.

        L (int or array): The multipole moment.
        M (int): The angular Fourier mode.
        psi (array): The psi values.
        """
        isscalar = np.isscalar(L)
        if isscalar:
            L = np.array([L])

        if np.any(L>self.Lmax):
            raise ValueError('L={} is larger than Lmax={}'.format(L, self.Lmax))
        if np.any(L<0):
            raise ValueError('L={} is smaller than 0'.format(L))
        if M>self.Mmax:
            raise ValueError('M={} is larger than Mmax={}'.format(M, self.Mmax))
        if M<-self.Mmax:
            raise ValueError('M={} is smaller than -Mmax={}'.format(M, self.Mmax))
        
        out = []
        for _L in L:
            # Use the symmetry for M<0
            # G_{LM}(psi) = G_{L(-M)}(np.pi/2-psi)
            if M>0:
                o = np.interp(psi, self.psi, self.GLMdata[(_L, M)])
            else:
                o = np.interp(np.pi/2-psi, self.psi, self.GLMdata[(_L, -M)])
            out.append(o)
        out = np.array(out)

        if isscalar:
            out = out[0]

        return out

class FastNaturalComponents:
    """
    Calculate the natural components using 2dfftlog.

    Supported projection of shear:
    - x: x-projection
    - cent: centroid-projection
    - ortho: orthocenter-projection
    """
    def __init__(self, Lmax, Mmax, bispectrum=None, verbose=True, config_bin=None):
        """
        Parameters
        ----------
        Lmax (int): The maximum multipole moment.
        Mmax (int): The maximum angular Fourier mode.
        bispectrum (Bispectrum, optional): The bispectrum object. Defaults to None.
        verbose (bool, optional): Whether to print the progress. Defaults to True.
        config_bin (dict, optional): The configuration for binning. Defaults to None.

        Notes on config_bin
        - auto (bool): Whether to automatically set ell12min, ell12max, and nell12bin. Defaults to True.
        - ell12min (float): The minimum value of ell12. Defaults to None. If auto is False, this must be specified.
        - ell12max (float): The maximum value of ell12. Defaults to None. If auto is False, this must be specified.
        - nell12bin (int): The number of bins for ell12. Defaults to 200.
        """
        # general setup
        self.verbose = verbose

        # initialize Lmax, Mmax
        self.Lmax = Lmax
        self.Mmax = Mmax

        # instantiate GLM calculator
        self.GLM = GLMCalculator(Lmax, Mmax, verbose=self.verbose)

        # 2DFFTLog config
        self.config_fftlog = {'nu1':1.01, 'nu2':1.01, 'N_pad':0}

        # set bispectrum
        if bispectrum is not None:
            self.set_bispectrum(bispectrum)

        # set bin
        self.config_bin = {'auto':True, 'ell12min':None, 'ell12max':None, 'nell12bin':150}
        if config_bin is not None:
            self.config_bin.update(config_bin)

        # flag for recomputation
        self.has_changed = True

    def set_bispectrum(self, bispectrum, **args):
        """
        Set and compute the bispectrum multipoles.

        bispectrum (Bispectrum): The bispectrum object.
        args (dict): The arguments for bispectrum.decompose.
        """
        # update bispectrum multipole
        self.bispectrum = bispectrum
        self.bispectrum.decompose(self.Lmax, **args)
        self.set_bin()
        self.has_changed = True

    def set_bin(self):
        """
        Set the bin.
        """
        if self.config_bin['auto']:
            ell12min = self.bispectrum.ell12min
            ell12max = self.bispectrum.ell12max
        else:
            ell12min = self.config_bin['ell12min']
            ell12max = self.config_bin['ell12max']
        nell12bin= self.config_bin['nell12bin']
        self.ell1 = self.ell2 = np.logspace(np.log10(ell12min), np.log10(ell12max), nell12bin)
        self.ELL1, self.ELL2 = np.meshgrid(self.ell1, self.ell2, indexing='ij')
        self.ELL  = np.sqrt(self.ELL1**2 + self.ELL2**2)
        self.PSI = np.arctan2(self.ELL2, self.ELL1)
    
    def HM(self, M, ell, psi, Lmax=None, Lmin=0):
        """
        Compute H_M(l1, l2 = \\sum_L (-1)^L * G_LM * b_L(l1, l2).

        M (int): The angular Fourier mode.
        ell (array): The ell values.
        psi (array): The psi values.
        Lmax (int, optional): The maximum value of L. Defaults to None.
        """
        # Get Lmax
        if Lmax is None:
            Lmax = self.Lmax

        # Sum up GLM*bL over L
        L = np.arange(Lmin, Lmax+1)
        GLM = self.GLM(L, M, self.PSI)
        bL = self.bispectrum.kappa_bispectrum_multipole(L, self.ELL, self.PSI)
        HM = np.sum(((-1)**(L+1)*GLM.T*bL.T).T, axis=0)
        return HM

    def GammaM(self, i, M, HM=None, dlnt=None, **args):
        """
        Compute Gamma^(M).
        
        i (int): The index of the natural component.
        M (int): The angular Fourier mode.
        HM (array, optional): The HM array. Defaults to None.
        Lmax (int, optional): The maximum value of L. Defaults to None.
        """
        # Compute HM
        if HM is None:
            HM = self.HM(M, self.ELL, self.PSI, **args)

        # transpose for component 3
        if i == 1:
            HM = HM.T

        # Get (n,m) from M.
        m, n = [(M-3,-M-3), (M-1,-M-1), (M+1,-M-3), (M-3,-M+1)][i]

        # Compute F_M using 2DFFTLog
        tb  = twobessel.two_Bessel(self.ell1, self.ell2, HM*self.ELL1**2*self.ELL2**2, **self.config_fftlog)
        if dlnt is None:
            self.t1, self.t2, GM = tb.two_Bessel(np.abs(m), np.abs(n))
        else:
            self.t1, self.t2, GM = tb.two_Bessel_binave(np.abs(m), np.abs(n), dlnt, dlnt)

        # Apply (-1)**m and (-1)**n 
        # These originate to J_m(x) = (-1)^m J_{-m}(x)
        if m < 0:
            GM *= (-1.)**m
        if n < 0:
            GM *= (-1.)**n
        GM /= (2*np.pi)**3

        self.T1, self.T2 = np.meshgrid(self.t1, self.t2, indexing='ij')

        # return
        return GM

    def update(self, indices=[0,1,2,3], Mmax=None, Lmax=None, dlnt=None):
        """
        Updates table.

        Mmax (int, optional): The maximum value of M. Defaults to None.
        Lmax (int, optional): The maximum value of L. Defaults to None.
        """
        # for test
        if Mmax is None:
            Mmax = self.Mmax
        if Lmax is None:
            Lmax = self.Lmax

        if isinstance(indices, int):
            indices = [indices]

        # 1. computes the kernel function GammaM
        self.tabGM = dict()
        pbar = tqdm(range(Mmax+1), desc='[GammaM]', disable=not self.verbose)
        for M in pbar:
            pbar.set_postfix({'M':M})
            HM = self.HM(M, self.ELL, self.PSI, Lmax=Lmax)
            for i in indices:
                self.tabGM[(i, M)] = self.GammaM(i, M, HM, dlnt=dlnt)

        self.has_changed = False

    def Gamma(self, i, phi, Mmax=None, projection='x'):
        """
        Compute Gamma(t1, t2, dphi)

        i (int): The index of the natural component.
        phi (float): The value of phi.
        Mmax (int, optional): The maximum value of M. Defaults to None.
        projection (str, optional): The projection shear. Defaults to 'x'.
        """
        # for test
        if Mmax is None:
            Mmax = self.Mmax

        # compute Gamma^0(t1, t2, dphi)
        Gamma = np.zeros(self.T1.shape, dtype=np.complex128)
        for M in range(-Mmax, Mmax+1):
            key = [(0, abs(M)), 
                   (1, abs(M)) if M>=0 else (2, abs(M)), 
                   (2, abs(M)) if M>=0 else (1, abs(M)), 
                   (3, abs(M))][i]
            GM = self.tabGM[key]
            GM = GM if M>=0 else GM.T
            Gamma += GM * np.exp(1j*M*phi)
        Gamma *= 1/(2*np.pi)

        # projection conversion
        Gamma *= self.projection_factor(i, self.T1, self.T2, phi, projection)

        # return
        return Gamma

    def Gamma0(self, phi, Mmax=None, projection='x'):
        """
        Compute Gamma^0(t1, t2, dphi)

        phi (float): The value of phi.
        """
        return self.Gamma(0, phi, Mmax=Mmax, projection=projection)
    
    def Gamma_treecorr(self, i, r, u, v, Mmax=None, projection='x', method='sdi', skip=1):
        """
        Compute Gamma^0(r, u, v) with treecorr convention.

        i (int): The index of the natural component.
        r (array): The value of r.
        u (float): The value of u.
        v (float): The value of v.
        Mmax (int, optional): The maximum value of M. Defaults to None.
        method (str, optional): The interpolation method. Defaults to 'sdi'.
        skip (int, optional): The skip factor for interpolation. Defaults to 1.
        """

        # Compute t1, t2, phi
        t1, t2, phi = trigutils.ruv_to_x1x2phi(r, u, v)

        # Compute Gamma0 without prefactor
        gamma0 = self.Gamma(i, phi, Mmax=Mmax, projection='x')

        # Interpolate
        if method == 'sdi':
            logx = np.log10(self.t1)
            f = (self.T1*self.T2)**0.5*gamma0
            f = sdi(logx[::skip], logx[::skip], f[::skip, ::skip])
            gamma0 = f(np.log10(t1), np.log10(t2)) / (t1*t2)**0.5
        elif method == 'rgi':
            # This gives artificial oscillations
            logx = np.log10(self.t1)
            f = (self.T1*self.T2)**0.5*gamma0
            f = rgi((logx[::skip], logx[::skip]), f[::skip, ::skip], method='linear')
            gamma0 = f((np.log10(t1), np.log10(t2))) / (t1*t2)**0.5

        # Compute and multiply prefactor
        gamma0 *= self.projection_factor(i, t1, t2, phi, projection)

        return gamma0

    def Gamma0_treecorr(self, r, u, v, Mmax=None, projection='x', method='sdi'):
        """
        Compute Gamma^0(r, u, v) with treecorr convention.

        r (array): The value of r.
        u (float): The value of u.
        v (float): The value of v.
        Mmax (int, optional): The maximum value of M. Defaults to None.
        projection (str, optional): The projection shear. Defaults to 'x'.
        method (str, optional): The interpolation method. Defaults to 'rgi'.
        """
        return self.Gamma_treecorr(0, r, u, v, Mmax=Mmax, projection=projection, method=method)

    # multiplicative phase factor to convert between different projections
    def projection_factor(self, i, t1, t2, phi, projection='x'):
        """
        Compute the projection factor.

        i (int): The index of the natural component.
        t1 (array): The value of t1.
        t2 (array): The value of t2.
        phi (float): The value of phi.
        projection (str, optional): The projection shear. Defaults to 'x'.
        """
        # Compute projection factor
        if projection == 'x':
            factor = 1
        elif projection == 'cent':
            factor = x2cent(i, t1, t2, phi)
        elif projection == 'ortho':
            factor = x2ortho(i, t1, t2, phi)
        else:
            raise ValueError('Error: projection={} is not expected'.format(projection))

        # return
        return factor

# phase factors to convert between different projections
def x2ortho(i, t1, t2, phi):
    # Compute prefactor
    sin2pb, cos2pb = sincos2angbar(np.arctan2(t2, t1), np.pi-phi)
    if i==0 or i==1 or i==2:
        out = cos2pb - 1j*sin2pb
    elif i==3:
        out = cos2pb + 1j*sin2pb
    return out

def ortho2cent(i, t1, t2, phi):
    t1, t2, t3 = trigutils.x1x2phi_to_x1x2x3(t1, t2, phi)

    def temp(t1, t2, t3):
        phi3 = np.arccos( (t1**2+t2**2-t3**2)/2/t1/t2 )
        cos2psi = ((t2**2-t1**2)**2 - 4*t1**2*t2**2*np.sin(phi3)**2)/4.0
        sin2psi = (t2**2-t1**2) * t1*t2 * np.sin(phi3)
        norm = np.sqrt(cos2psi**2 + sin2psi**2)
        exp2psi = cos2psi/norm + 1j*sin2psi/norm
        return exp2psi

    exp2psi3 = temp(t1, t2, t3)
    exp2psi1 = temp(t2, t3, t1)
    exp2psi2 = temp(t3, t1, t2)

    out = 1
    for j, phase in enumerate([1.0, exp2psi1, exp2psi2, exp2psi3]):
        if j == i:
            out *= phase
        else:
            out *= np.conj(phase)

    return out

def x2cent(i, t1, t2, phi):
    # Equations between Eq. (15) and (16) 
    # of https://arxiv.org/abs/2309.08601
    v = t1+t2*np.exp(-1j*phi)
    q1 = v/np.conj(v)
    v = -2*t1+t2*np.exp(-1j*phi)
    q2 = v/np.conj(v)
    v = t1-2*t2*np.exp(-1j*phi)
    q3 = v/np.conj(v)
    
    if i==0:
        return q1*q2*q3 * np.exp(3j*phi)
    elif i==1:
        return np.conj(q1)*q2*q3 * np.exp(1j*phi)
    elif i==2:
        return q1*np.conj(q2)*q3 * np.exp(3j*phi)
    elif i==3:
        return q1*q2*np.conj(q3) * np.exp(-1j*phi)
    else:
        raise ValueError('Error: i={} is not expected'.format(i))