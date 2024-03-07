#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/03/06 13:50:33

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
        # FFT grid in Fourier space
        self.ELL1, self.ELL2 = np.meshgrid(self.ell1, self.ell2, indexing='ij')
        self.ELL  = np.sqrt(self.ELL1**2 + self.ELL2**2)
        self.PSI = np.arctan2(self.ELL2, self.ELL1)
        # FFT grid in real space
        self.t1, self.t2 = 1/self.ell1[::-1], 1/self.ell2[::-1]
        self.T1, self.T2 = np.meshgrid(self.t1, self.t2, indexing='ij')
    
    def HM(self, M, ell, psi, bL=None, Lmin=None, Lmax=None):
        """
        Compute H_M(l1, l2 = \\sum_L (-1)^L * G_LM * b_L(l1, l2).

        M (int): The angular Fourier mode.
        ell (array): The ell values.
        psi (array): The psi values.
        bL (array): The bispectrum multipole. Defaults to None.
                    If None, it is computed using self.bispectrum.kappa_bispectrum_multipole.
                    By supplying bL, you can avoid recomputation of bL.
        Lmin (int, optional): The minimum value of L. Defaults to None.
        Lmax (int, optional): The maximum value of L. Defaults to None.
        """
        # Get bispectrum multipole indices, L array
        Lmin = Lmin or 0
        Lmax = Lmax or self.Lmax
        L = np.arange(Lmin, Lmax+1)

        # Get bispectrum multipole
        if bL is None:
            bL = self.bispectrum.kappa_bispectrum_multipole(L, self.ELL, self.PSI)

        # Sum up GLM*bL over L
        GLM = self.GLM(L, M, self.PSI)
        HM = np.sum(((-1)**(L+1)*GLM.T*bL.T).T, axis=0)
        return HM

    def __init_kernel_table(self, Mmax=None, Lmin=None, Lmax=None):
        """
        Initialize kernel table.

        Mmax (int, optional): The maximum value of M. Defaults to None.
        """
        # natural-component multipole indices
        Mmax = Mmax or self.Mmax
        M = np.arange(Mmax+1)

        # bispectrum multipole indices
        Lmin = Lmin or 0
        Lmax = Lmax or self.Lmax
        L = np.arange(Lmin, Lmax+1)

        # bispectrum multipole
        bL = self.bispectrum.kappa_bispectrum_multipole(L, self.ELL, self.PSI)

        # initialize table
        self.tabHM = dict()
        for _ in tqdm(M, desc='[HM]', disable=not self.verbose):
            HM = self.HM(_, self.ELL, self.PSI, bL=bL, Lmin=Lmin, Lmax=Lmax)
            self.tabHM[_] = HM

        # update flag
        self.has_changed = False
        
    def GammaM_on_grid(self, mu, M, dlnt=None):
        """
        Compute Gamma^(M).
        
        mu (array): The index of the natural component.
        M (array): The angular Fourier mode.
        dlnt (float, optional): The bin width for t1 and t2. Defaults to None. 
                                When None, dlnt=0, i.e. no bin averaging effect

        GammaM is computed on FFT grid
        The output shape will be (mu.size, M.size, self.t1.size, self.t2.size)
        """
        # get kernel
        if self.has_changed:
            self.__init_kernel_table()

        # casting to array
        if np.isscalar(mu):
            mu = np.array([mu])
        if np.isscalar(M):
            M = np.array([M])

        # Some GammaM are degenerating 
        # and we want to avoid recomputation for GammaM. 
        # Here we prepare the request for which GammaM 
        # to compute. We later assign the results.
        request = dict()
        for _mu in mu:
            for _M in M:
                if _M<0:
                    continue
                if _M not in request:
                    request[_M] = [_mu]
                else:
                    request[_M].append(_mu)

        # Compute
        tabGM = dict()
        for _M, mu_list in request.items():
            # Initialize 2D-FFTLog, this is shared for all mu to speed up.
            HM = self.tabHM[_M]
            tb  = twobessel.two_Bessel(self.ell1, self.ell2, HM*self.ELL1**2*self.ELL2**2, **self.config_fftlog)
            # Loop over mu
            for _mu in mu_list:
                # Get (n,m) from M.
                m, n = [(_M-3,-_M-3), (-_M-1,_M-1), (_M+1,-_M-3), (_M-3,-_M+1)][_mu]
                if dlnt is None:
                    # compute GammaM on FFT grid
                    GM = tb.two_Bessel(np.abs(m), np.abs(n))[2]
                elif dlnt is not None:
                    # compute GammaM on FFT grid with bin-averaging effect
                    GM = tb.two_Bessel_binave(np.abs(m), np.abs(n), dlnt, dlnt)[2]
                # Apply (-1)**m and (-1)**n
                # These originate to J_m(x) = (-1)^m J_{-m}(x)
                GM *= (-1.)**m if m<0 else 1
                GM *= (-1.)**n if n<0 else 1
                # normalization
                GM /= (2*np.pi)**3

                # store
                if _mu == 1:
                    tabGM[(_mu, -_M)] = GM.T
                else:
                    tabGM[(_mu, _M)] = GM

        # Assign
        GM = []
        for _mu in mu:
            _ = []
            for _M in M:
                if _mu == 0 and _M>=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 0 and _M<0:
                    _.append(tabGM[(_mu, -_M)].T)
                if _mu == 1 and _M>0:
                    _.append(tabGM[(_mu, -_M)].T)
                if _mu == 1 and _M<=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 2 and _M>=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 2 and _M<0:
                    _.append(tabGM[(3, -_M)].T)
                if _mu == 3 and _M>=0:
                    _.append(tabGM[(_mu, _M)])
                if _mu == 3 and _M<0:
                    _.append(tabGM[(2, -_M)].T)
            GM.append(_)
        GM = np.array(GM)

        # return
        return GM

    def GammaM_on_bin(self, mu, M, t1, t2, dlnt=None):
        """
        Compute Gamma^(M).
        
        mu (int): The index of the natural component.
        M (int): The angular Fourier mode.
        dlnt (float, optional): The bin width for t1 and t2. Defaults to None.
        t1 (array, optional): The value of t1. Defaults to None.
        t2 (array, optional): The value of t2. Defaults to None.
        Lmax (int, optional): The maximum value of L. Defaults to None.

        GammaM is computed on user-defined bin
        The output shape is (mu.size, M.size) + t1.shape
        """
        # get kernel
        if self.has_changed:
            self.__init_kernel_table()

        # casting to array
        if np.isscalar(mu):
            mu = np.array([mu])
        if np.isscalar(M):
            M = np.array([M])

        # Compute
        tabGM = dict()
        for _M in M:
            # Initialize 2D-FFTLog, this is shared for all mu to speed up.
            if _M >= 0:
                HM = self.tabHM[_M]
            else:
                HM = self.tabHM[-_M].T
            tb  = twobessel.two_Bessel(self.ell1, self.ell2, HM*self.ELL1**2*self.ELL2**2, **self.config_fftlog)
            # Loop over mu
            for _mu in mu:
                # Get (n,m) from M.
                m, n = [(_M-3,-_M-3), (-_M-1,_M-1), (_M+1,-_M-3), (_M-3,-_M+1)][_mu]
                if dlnt is None:
                    # compute GammaM on user-defined grid
                    GM = tb.two_Bessel_on_bin(np.abs(m), np.abs(n), t1, t2)[2]
                if dlnt is not None:
                    # compute GammaM on user-defined grid with bin-averaging effect
                    GM = tb.two_Bessel_binave_on_bin(np.abs(m), np.abs(n), t1, t2, dlnt, dlnt)[2]
                # Apply (-1)**m and (-1)**n
                # These originate to J_m(x) = (-1)^m J_{-m}(x)
                GM *= (-1.)**m if m<0 else 1
                GM *= (-1.)**n if n<0 else 1
                # normalization
                GM /= (2*np.pi)**3

                # store
                if _mu == 1:
                    tabGM[(_mu, -_M)] = GM.T # This is wrong, need to be fixed
                else:
                    tabGM[(_mu, _M)] = GM

        # Assign
        GM = []
        for _mu in mu:
            _ = []
            for _M in M:
                _.append(tabGM[(_mu, _M)])
            GM.append(_)
        GM = np.array(GM)

        # return
        return GM

    def GammaM(self, mu, M, t1=None, t2=None, dlnt=None):
        if t1 is None and t2 is not None:
            raise ValueError('Error: t1 is None but t2 is not None')
        if t1 is not None and t2 is None:
            raise ValueError('Error: t1 is not None but t2 is None')
        if t1 is None or t2 is None:
            GM = self.GammaM_on_grid(mu, M, dlnt=dlnt)
        else:
            GM = self.GammaM_on_bin(mu, M, t1, t2, dlnt=dlnt)
        return GM

    def Gamma(self, mu, phi, t1=None, t2=None, Mmax=None, dlnt=None, projection='x'):
        """
        Compute Gamma_mu(t1, t2, dphi)

        mu (int): The index of the natural component.
        phi (float): The value of phi.
        t1 (array, optional): The value of t1. Defaults to None.
        t2 (array, optional): The value of t2. Defaults to None.
        Mmax (int, optional): The maximum value of M. Defaults to None.
        projection (str, optional): The projection shear. Defaults to 'x'.
        """
        Mmax = Mmax or self.Mmax

        # casting to array
        if np.isscalar(mu):
            mu = np.array([mu])
        if np.isscalar(phi):
            phi = np.array([phi])

        # compute multipoles
        M       = np.arange(-Mmax, Mmax+1)
        GM      = self.GammaM(mu, M, t1=t1, t2=t2, dlnt=dlnt)

        # resummation
        if t1 is not None and t2 is not None:
            GM      = np.reshape(GM, (len(mu), M.size, -1))
            expMphi = np.exp(1j*M[:,None]*np.reshape(phi,-1))
            Gamma   = np.einsum('imk,mk->ik', GM, expMphi)/(2*np.pi)
            Gamma   = np.reshape(Gamma, (len(mu),)+t1.shape)
            Gamma  *= self.projection_factor(mu, t1, t2, phi, projection)
        else:
            expMphi = np.exp(1j*M[:,None]*phi)
            Gamma   = np.einsum('im...,mk->ik...', GM, expMphi)/(2*np.pi)
            Gamma  *= self.projection_factor(mu, self.T1, self.T2, phi, projection)

        return Gamma
    
    # multiplicative phase factor to convert between different projections
    def projection_factor(self, i, phi, t1=None, t2=None, projection='x'):
        """
        Compute the projection factor.

        i (int): The index of the natural component.
        t1 (array): The value of t1.
        t2 (array): The value of t2.
        phi (float): The value of phi.
        projection (str, optional): The projection shear. Defaults to 'x'.
        """
        if t1 is None:
            t1 = self.T1
        if t2 is None:
            t2 = self.T2

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