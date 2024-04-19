#!/usr/bin/env python
'''
Author     : Sunao Sugiyama 
Last edit  : 2024/04/08 14:06:25

Description:
coupling.py contains classes for 
the computing multipole coupling functions 
'''
import numpy as np
from scipy.special import eval_legendre
import pandas as pd
import os
import json
# fastnc modules
from .utils import sincos2angbar, npload_lock, npsavez_lock
from .integration import aint

def get_cache_dir():
    # first we look for the environmental variable
    cache_dir = os.environ.get('FASTNC_CACHE_DIR')
    if cache_dir is not None:
        cache_dir = os.path.expanduser(cache_dir)
        return cache_dir
    # finally we use the default directory
    cache_dir = '~/.fastnc'
    # Expand ~ to the user's home directory
    cache_dir = os.path.expanduser(cache_dir)
    return cache_dir

def mkcachedir_if_not_exists(dirname):
    if not os.path.exists(dirname):
        print('Creating cache directory: {}'.format(dirname))
        os.makedirs(dirname, exist_ok=True)

class CacheManager:
    """
    This class manages the chache
    """
    entry_id_keys = ['name', 'Npsi', 'tol']
    def __init__(self):
        self.cache_dir = get_cache_dir()
        mkcachedir_if_not_exists(self.cache_dir)
        self.databasename = os.path.join(self.cache_dir, 'database.csv')
        self.df = self.__load_database()

    def __load_database(self):
        # Load the CSV file into memory if it exists, otherwise initialize an empty dataframe
        if os.path.exists(self.databasename):
            return pd.read_csv(self.databasename)
        else:
            return pd.DataFrame(columns=self.entry_id_keys+['filename'])

    def get_entry(self, entry_id):
        sel = [self.df[key] == entry_id[key] for key in self.entry_id_keys]
        sel = np.all(sel, axis=0)
        return self.df[sel]

    def in_database(self, entry_id):
        return not self.get_entry(entry_id).empty

    def get_entry_filename(self, entry_id):
        existing_entry = self.get_entry(entry_id)
        return os.path.join(self.cache_dir, existing_entry['filename'].values[0])

    def register_entry(self, entry_id):
        if self.in_database(entry_id):
            return 0

        # If entry does not exist, generate a new unique filename
        new_filename = self.__generate_unique_filename()
        
        # Add a new row to the dataframe with the provided Npsi, tol, and generated filename
        entry_id['filename'] = [new_filename]
        new_row = pd.DataFrame(entry_id)
        self.df = pd.concat([self.df, new_row], ignore_index=True)

        # Write the updated dataframe to the CSV file
        self.write()

    def __generate_unique_filename(self):
        # Generate a new filename that does not already exist in the dataframe
        i = 1
        while True:
            new_filename = f"coupling-{i}.npz"
            if new_filename not in self.df['filename'].values:
                return new_filename
            i += 1

    def write(self):
        # Write the current dataframe to the CSV file
        self.df.to_csv(self.databasename, index=False)

class ModeCouplingFunctionBase:
    r"""
    Compute and hold the multipole mode-coupling function between real
    and Fourier spaces.

    Parameters:
        Lmax (int)    : Maximum multipole moment.
        Mmax (int)    : Maximum angular Fourier mode.
        Npsi (int)    : Number of psi bins. Npsi is forced to be odd to capture the 
                        exact pi/2 point.
        tol (float)   : Tolerance for the numerical integration.
        verbose (bool): Whether to print verbose output.
        cache (bool)  : Whether to use cache.
    """

    # name of the mode coupling function
    name = 'ModeCouplingFunctionBase'
    
    def __init__(self, Lmax, Mmax, Npsi=200, tol=1e-5, verbose=True, cache=True):
        self.Lmax = Lmax
        self.Mmax = Mmax
        self.Npsi = int((Npsi//2)*2+1)
        self.psi = np.linspace(0, np.pi/2, Npsi)
        self.tol = tol
        self.verbose = verbose
        self.cache = cache

        # mode coupling function data
        self.data = dict()
        # load cache if exists
        self.load_cache() if self.cache else None
        # compute mode coupling function
        has_changed = self.compute()
        # save cache
        if self.cache and has_changed:
            self.save_cache()

    def compute(self):
        """
        This will compute the mode coupling function
        which was not found in the cache.

        Returns:
            bool: Whether the data has changed.
        """
        raise NotImplementedError

    def _get_id(self):
        return {'name':self.name, 'Npsi':self.Npsi, 'tol':self.tol}

    def save_cache(self):
        """
        Save the data to the cache.
        """
        # load database and register the data info in the database
        database = CacheManager()
        database.register_entry(self._get_id())
        # save the data to the cache
        filename = database.get_entry_filename(self._get_id())
        # because the npz file only accepts string keys,
        # we convert the keys to string using json encoding
        cache = {json.dumps(key): value for key, value in self.data.items()}
        npsavez_lock(filename, cache, suffix='.npz')

    def load_cache(self):
        """
        Load the data from the cache.
        """
        # load database
        database = CacheManager()
        if database.in_database(self._get_id()):
            # load the data from the cache
            filename = database.get_entry_filename(self._get_id())
            print(f'Loading cache from cache at {filename}') if self.verbose else None
            cache = npload_lock(filename, suffix='.npz')
            # decoding
            # because json encodes tuple into string which looks like a list,
            # we convert the decoded string into tuple by hand since list
            # is not hashable: it cannot be used for the key of dict.
            self.data = {tuple(json.loads(key)): value for key, value in dict(cache).items()}

    def __call__(self, L, M, psi):
        """
        Compute GLM for given L, M, and psi.

        L (array): The multipole moment.
        M (array): The angular Fourier mode.
        psi (array): The psi values.
        """
        pass

class MCF222LegendreFourier(ModeCouplingFunctionBase):
    r"""
    Mode coupling function for spin-2, spin-2, spin-2 correlation function
    and decomposition using Legendre polynomials and Fourier modes
    for Fourier-space and real-space respectively.

    .. math:
        G_LM(psi) = 4\\pi \\int_0^{\\pi} dx P_L(\\cos(x)) \\cos[2\\bar\\beta(\\psi, x) + M x]
    """
    __doc__ += ModeCouplingFunctionBase.__doc__
    
    # name of the mode coupling function
    name = 'MCF222LegendreFourier'

    def _integrand(self, x, psi, L, M):
        x, psi = np.meshgrid(x, psi, indexing='ij')
        sin2bb, cos2bb = sincos2angbar(psi, x)
        out = 4*np.pi*eval_legendre(L, np.cos(x)) * (cos2bb*np.cos(M*x) - sin2bb*np.sin(M*x))
        return out
        
    def compute(self, correct_bias=True):
        has_changed = False
        # prepare todo list
        todo = []
        for L in range(self.Lmax+1):
            for M in range(self.Mmax+1):
                todo.append([L, M])
        # compute GLM
        for L, M in todo:
            print(f'\r(L,M) = {(L,M)}/{len(todo)}', end='') if self.verbose else None
            # skip if the data already exists
            # in the cache
            if (L,M) in self.data:
                continue
            args = {'L':L, 'M':M, 'psi':self.psi}
            o, c = aint(self._integrand, 0, np.pi, 2, tol=self.tol, **args)
            self.data[(L, M)] = o
            has_changed = True
        # Correction
        # G_LM(psi) is exactly zero for L<M and psi<=pi/4. 
        # However, the numerical integration may give a non-zero value.
        # From my experience, the same amount of error is also present in
        # G_ML if G_LM is biased. Hence we subtract the error in G_LM(psi<=pi/4)
        # from G_LM(psi) and G_ML(psi).
        if correct_bias:
            for (L,M), data in self.data.items():
                if L>=M:
                    continue
                # estimate the bias in G_LM(psi<=np.pi/4)
                bias = np.mean(data[self.psi<=np.pi/4])
                # subtract the bias
                self.data[(L,M)] -= bias
                if (M,L) in self.data:
                    self.data[(M,L)] -= bias
                has_changed = True
        return has_changed

    def __call__(self, L, M, psi):
        Lisscalar = np.isscalar(L)
        if Lisscalar:
            L = np.array([L])
        Misscalar = np.isscalar(M)
        if Misscalar:
            M = np.array([M])

        if np.any(L>self.Lmax):
            raise ValueError('L={} is larger than Lmax={}'.format(L, self.Lmax))
        if np.any(L<0):
            raise ValueError('L={} is smaller than 0'.format(L))
        if np.any(M>self.Mmax):
            raise ValueError('M={} is larger than Mmax={}'.format(M, self.Mmax))
        if np.any(M<-self.Mmax):
            raise ValueError('M={} is smaller than -Mmax={}'.format(M, self.Mmax))
        
        # collect todo
        todo = []
        for _L in L:
            for _M in M:
                todo.append([_L, _M])

        out = []
        for _L, _M in todo:
            # Use the symmetry for M<0
            # G_{LM}(psi) = G_{L(-M)}(np.pi/2-psi)
            if _M>0:
                o = np.interp(psi, self.psi, self.data[(_L, _M)])
            else:
                o = np.interp(np.pi/2-psi, self.psi, self.data[(_L, -_M)])
            out.append(o)
        out = np.array(out).reshape(L.shape+M.shape+psi.shape)

        if Lisscalar and Misscalar:
            out = out[0,0]
        elif Lisscalar:
            out = out[0]
        elif Misscalar:
            out = out[:,0]

        return out

class MCF222FourierFourier(ModeCouplingFunctionBase):
    r"""
    Mode coupling function for spin-2, spin-2, spin-2 correlation function
    and decomposition using Fourier modes both
    for Fourier-space and real-space respectively.

    .. math:
        G_LM(\\psi) = 4\\pi \\int_0^{\\pi} dx \\cos[2\\bar\\beta(\\psi, x) + (M-L) x]

    This can be parametrized by a single integer K=M-L.
    
    ..math::
        G_K(\\psi) = 4\\pi \\int_0^{\\pi} dx \\cos[2\\bar\\beta(\\psi, x) + K x]

    """
    __doc__ += ModeCouplingFunctionBase.__doc__

    # name of the mode coupling function
    name = 'MCF222FourierFourier'
    
    def _integrand(self, x, psi, K):
        x, psi = np.meshgrid(x, psi, indexing='ij')
        sin2bb, cos2bb = sincos2angbar(psi, x)
        out = 4*np.pi*(cos2bb*np.cos(K*x) - sin2bb*np.sin(K*x))
        return out
        
    def compute(self, correct_bias=True):
        has_changed = False
        self.Kmax = self.Mmax + self.Lmax
        # compute GK
        for K in range(self.Kmax+1):
            print(f'\rK = {K}/{2*self.Kmax+1}', end='') if self.verbose else None
            # skip if the data already exists
            # in the cache
            if K in self.data:
                continue
            args = {'K':K, 'psi':self.psi}
            o, c = aint(self._integrand, 0, np.pi, 2, tol=self.tol, **args)
            self.data[K] = o
            has_changed = True
        # Correction?
        return has_changed

    def __call__(self, L, M, psi):
        Lisscalar = np.isscalar(L)
        if Lisscalar:
            L = np.array([L])
        Misscalar = np.isscalar(M)
        if Misscalar:
            M = np.array([M])

        if np.any(L>self.Lmax):
            raise ValueError('L={} is larger than Lmax={}'.format(L, self.Lmax))
        if np.any(L<-self.Lmax):
            raise ValueError('L={} is smaller than -Lmax={}'.format(L, self.Lmax))
        if np.any(M>self.Mmax):
            raise ValueError('M={} is larger than Mmax={}'.format(M, self.Mmax))
        if np.any(M<-self.Mmax):
            raise ValueError('M={} is smaller than -Mmax={}'.format(M, self.Mmax))
        
        # collect todo
        todo = []
        for _L in L:
            for _M in M:
                todo.append([_L, _M])

        out = []
        for _L, _M in todo:
            # Use the symmetry for M<0
            # G_{K}(psi) = G_{-K}(np.pi/2-psi)
            _K = _M - _L
            if _K>0:
                o = np.interp(psi, self.psi, self.data[_K])
            else:
                o = np.interp(np.pi/2-psi, self.psi, self.data[-_K])
            out.append(o)
        out = np.array(out).reshape(L.shape+M.shape+psi.shape)

        if Lisscalar and Misscalar:
            out = out[0,0]
        elif Lisscalar:
            out = out[0]
        elif Misscalar:
            out = out[:,0]

        return out
