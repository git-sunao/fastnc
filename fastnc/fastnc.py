"""
This is the module of fastnc, which calculate the natural components 
of 3PCF using multipole decomposition and 2DFFTLog.

Author: Sunao Sugiyama
Last edit: 2023/11/16
"""
import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import RegularGridInterpolator as rgi
from tqdm import tqdm
from glob import glob
import pandas as pd
import os, pickle
# fastnc modules
from . import twobessel
from . import utils
from . import trigutils
from .multipole import BispectrumMultipoleCalculator

def sincos2angbar(psi, delta):
    cos2b = np.cos(delta) + np.sin(2*psi)
    sin2b = np.cos(2*psi) * np.sin(delta)
    norm  = np.sqrt(cos2b**2 + sin2b**2)
    return sin2b/norm, cos2b/norm

def _save_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

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
        G_LM(psi) = 2 \int_0^{\pi} dx P_L(\cos(x)) \cos[2\bar\beta(\psi, x) + M x]

    Parameters
    ----------
    Lmax : int
        Maximum multipole moment.
    Max  : int
        Maximum angular Fourier mode.
    Npsi : int
        Number of psi bins.
    
    Attributes
    ----------
    Lmax : int
        Maximum multipole moment.
    Max  : int
        Maximum angular Fourier mode.
    psi  : array
        psi bins.
    """

    cachedir = os.path.join(os.path.dirname(__file__), 'cache')
    databasename = os.path.join(cachedir, 'GLMdatabase.csv')

    def __init__(self, Lmax, Mmax, Npsi=200, tol=1e-3, verbose=False, cache=True):
        self.Lmax = Lmax
        self.Mmax = Mmax
        self.psi  = np.linspace(0, np.pi/2, Npsi)
        self.tol  = tol

        # initialize GLMdata
        self.GLMdata = dict()

        # load GLMdata from cache if exists
        self.load_cache()

        # compute/update GLM 
        self.compute_GLM(self.tol, verbose=verbose)

        # save cache
        if cache:
            self.save_cache()

    def integrand(self, x, psi, L, M):
        x, psi = np.meshgrid(x, psi, indexing='ij')
        sin2bb, cos2bb = sincos2angbar(psi, x)
        out = 4*np.pi*eval_legendre(L, np.cos(x)) * (cos2bb*np.cos(M*x) - sin2bb*np.sin(M*x))
        return out
        
    def compute_GLM(self, tol, correct_bias=True, verbose=False):
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
            o, c = utils.aint(self.integrand, 0, np.pi, 2, tol=tol, **args)
            self.GLMdata[(L, M)] = o

        # Correction
        # G_LM(psi) is exactly zero for L<M and psi<=pi/4. 
        # However, the numerical integration may give a non-zero value.
        # From my experience, this amount of error is also present in
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
        _save_pickle(os.path.join(self.cachedir, filename), self.GLMdata)

    def load_cache(self):
        # load database
        database = GLMdatabase(self.databasename)

        # filename
        filename = database.find_entry(len(self.psi), self.tol)

        # load
        if filename is not None:
            print('Load GLMdata from cache: {}'.format(filename))
            self.GLMdata = _load_pickle(os.path.join(self.cachedir, filename))

    def __call__(self, L, M, psi):
        if L>self.Lmax:
            raise ValueError('L={} is larger than Lmax={}'.format(L, self.Lmax))
        if L<0:
            raise ValueError('L={} is smaller than 0'.format(L))
        if M>self.Mmax:
            raise ValueError('M={} is larger than Mmax={}'.format(M, self.Mmax))
        if M<-self.Mmax:
            raise ValueError('M={} is smaller than -Mmax={}'.format(M, self.Mmax))
        
        # Use the symmetry for M<0
        # G_{LM}(psi) = G_{L(-M)}(np.pi/2-psi)
        if M>0:
            return np.interp(psi, self.psi, self.GLMdata[(L, M)])
        else:
            return np.interp(np.pi/2-psi, self.psi, self.GLMdata[(L, -M)])
            
class FastNaturalComponentsCalcurator:
    def __init__(self, bispectrum, lmin, lmax, Lmax, Mmax, nbin=200):
        # initialize bins
        self.l12_1d = np.logspace(np.log10(lmin), np.log10(lmax), nbin)
        self.l1, self.l2 = np.meshgrid(self.l12_1d, self.l12_1d, indexing='ij')
        self.l   = np.sqrt(self.l1**2 + self.l2**2)
        self.psi = np.arctan2(self.l2, self.l1)

        # initialize Lmax, Mmax
        self.Lmax = Lmax
        self.Mmax = Mmax

        # initialize FMdata
        self.FMdata = dict()

        # instantiate GLM calculator
        self.GLM = GLMCalculator(Lmax, Mmax, verbose=True)

        # initialize bispectrum multipole
        self.set_bispectrum(bispectrum)

        # 2DFFTLog config
        self.fftlog_config = {'nu1':1.01, 'nu2':1.01, 'N_pad':0}

    def set_bispectrum(self, bispectrum):
        """
        Set and compute the bispectrum multipoles.
        """
        if not hasattr(self, 'bispectrum_multipole'):
            # initialize bispectrum multipole
            Lmax = self.Lmax
            lmin, lmax = self.l.min()/1.01, self.l.max()*1.01
            psimin = self.psi.min()/1.01
            self.bispectrum_multipole = BispectrumMultipoleCalculator(bispectrum, Lmax, lmin, lmax, psimin)
        else:
            # update bispectrum multipole
            self.bispectrum_multipole.set_bispectrum(bispectrum)
        
    def sumGLMbL(self, M, l, psi, Lmax=None):
        # Get Lmax
        if Lmax is None:
            Lmax = self.Lmax

        # Sum up GLM*bL over L
        sumGLMbL = 0
        for L in range(Lmax+1):
            # Compute GLM
            GLM = self.GLM(L, M, self.psi)

            # Compute bL
            bL = self.bispectrum_multipole(L, self.l, self.psi)

            # Add
            sumGLMbL += GLM*bL
        
        return sumGLMbL

    def FM(self, i, M, Lmax=None):
        # Compute sumGLMbL = \sum_L G_LM * b_L(l1, l2)
        sumGLMbL = self.sumGLMbL(M, self.l, self.psi, Lmax=Lmax)

        # Get (n,m) from M.
        m, n = [(M-3,-M-3), (M+1,-M-3), (M-3,-M+1), (M-1,-M+1)][i]

        # Compute F_M using 2DFFTLog
        tb  = twobessel.two_Bessel(self.l12_1d, self.l12_1d, sumGLMbL*self.l1**2*self.l2**2, **self.fftlog_config)
        x1, x2, FM = tb.two_Bessel(np.abs(m), np.abs(n))
        
        # Apply (-1)**m and (-1)**n 
        # These originate to J_m(x) = (-1)^m J_{-m}(x)
        if m < 0:
            FM *= (-1.)**m
        if n < 0:
            FM *= (-1.)**n

        # Transpose
        # Note that (l1, l2) is couple to (x2,x1), 
        # not to (x1, x2) which is convention of 2DFFTLog.
        FM = FM.T

        self.x1, self.x2 = np.meshgrid(x1, x2, indexing='ij')
        self.x12_1d = x1

        # return
        return FM

    def compute_FM(self, i, Lmax=None):
        # Initialize
        self.FMdata[i] = dict()

        # Compute FM
        for M in range(self.Mmax+1):
            FM = self.FM(i, M, Lmax=Lmax)
            self.FMdata[i][M] = FM

            if M != 0:
                self.FMdata[i][-M] = FM.T

    def Gamma0(self, dvarphi, multiply_prefactor=True):
        """
        Compute Gamma^0(x1, x2, dphi)

        Parameters
        ----------
        dvarphi : float
            The value of dvarphi.
        
        Returns
        -------
        Gamma0 : ndarray
            Gamma^0(x1, x2, dphi).
        """
        # compute phibar
        psi = np.arctan2(self.x2, self.x1)
        sin2pb, cos2pb = sincos2angbar(psi, dvarphi)

        # compute Gamma^0(x1, x2, dphi)
        gamma = np.zeros(psi.shape, dtype=np.complex128)
        for M, FM in self.FMdata[0].items():
            # Compute phase
            phase  = (-1.)**M * np.exp(1j*M*dvarphi)

            # add
            gamma+= FM * phase

        # multiply prefactor
        # This multiplicative factor can have very sharp features
        # in (l1,l2) grid and hence the interpolation may not work well.
        # We recommend one to interpolate Gamma^0(x1, x2, dphi) without
        # this prefactor and multiply it later.
        if multiply_prefactor:
            prefactor = -1/(2*np.pi)**4 * (cos2pb - 1j*sin2pb)
            gamma *= prefactor

        # return
        return gamma

    def Gamma0_treecorr(self, r, u, v, tid=0, multiply_prefactor=True):
        """
        Compute Gamma^0(r, u, v) with treecorr convention.

        Parameters
        ----------
        r : array
            The value of r.
        u : float
            The value of u.
        v : float
            The value of v.

        Returns
        -------
        Gamma0 : ndarray
            Gamma^0(r, u, v).
        """

        # Compute x1, x2, dvphi
        if tid == 0:
            x1, x2, dvphi = trigutils.ruv_to_x1x2dvphi(r, u, v)
        if tid == 1:
            x1, x2, dvphi = trigutils.ruv_to_x2x3dvphi(r, u, v)
        if tid == 2:
            x1, x2, dvphi = trigutils.ruv_to_x3x1dvphi(r, u, v)

        # Compute Gamma0 without prefactor
        gamma0 = self.Gamma0(dvphi, multiply_prefactor=False)

        # Interpolate
        logx = np.log10(self.x12_1d)
        f = rgi((logx, logx), (self.x1*self.x2)**0.5*gamma0, method='cubic')
        gamma0 = f((np.log10(x1), np.log10(x2))) / (x1*x2)**0.5

        # Compute and multiply prefactor
        if multiply_prefactor:
            sin2pb, cos2pb = sincos2angbar(np.arctan2(x2, x1), dvphi)
            prefactor = -1/(2*np.pi)**4 * (cos2pb - 1j*sin2pb)
            gamma0 *= prefactor

        return gamma0