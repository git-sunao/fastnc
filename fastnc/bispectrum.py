"""
Bispectrum module.

Author: Sunao Sugiyama
Last edit: 2024/01/17
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RegularGridInterpolator as rgi
from astropy.cosmology import wCDM, Planck18
from scipy.special import sici
from scipy.special import eval_legendre

from . import trigutils
from .halofit import Halofit
from .multipole import Multipole
from .utils import loglinear, edge_correction

wPlanck18 = wCDM(H0=Planck18.H0, Om0=Planck18.Om0, Ode0=Planck18.Ode0, w0=-1.0, meta=Planck18.meta, name='wPlanck18')

class BispectrumBase:
    def __init__(self, cosmo=None, zs=None, pzs=None):
        self.set_cosmology(cosmo or wPlanck18)
        self.set_source_distribution(zs or 1, pzs or 1)

    # setter
    def set_cosmology(self, cosmo):
        """
        cosmo (astropy.cosmology): cosmology
        """
        self.cosmo = cosmo

        # compute array of chi and z
        z   = np.linspace(0, 5, 100)
        chi = self.cosmo.comoving_distance(z).value * self.cosmo.h # Mpc/h

        # spline chi <-> z
        self.z2chi = ius(z, chi)
        self.chi2z = ius(chi, z)

    def set_source_distribution(self, zs, pzs, nzlbin=101):
        """
        Set source distribution.

        zs (array): redshift array
        pzs (array): probability distribution of source galaxies
        """

        if np.isscalar(zs):
            zl = np.linspace(0.0, zs, nzlbin)
            chil = self.z2chi(zl)
            chis = self.z2chi(zs)
            g = 1.-chil/chis
            self.zmax = zs
            self.z2g = ius(zl, g, ext=1)
            self.chi2g = ius(chil, g, ext=1)
        else:
            assert len(zs) == len(pzs), "zs and pzs must have the same length"
            zl = np.linspace(0, zs.max(), nzlbin)
            chis = self.z2chi(zs)
            chil = self.z2chi(zl)
            CHIS, CHIL = np.meshgrid(chis, chil, indexing='ij')

            # integrand
            I = np.zeros_like(CHIL, dtype=float)
            I = np.divide(CHIL, CHIS, out=I, where=CHIS >= CHIL)
            I = pzs*(1-I)

            g = np.trapz(I, chis, axis=1)/np.trapz(pzs, chis)
            self.z2g = ius(zl, g, ext=1)
            self.chi2g = ius(chil, g, ext=1)
            self.zmax = zs.max()

    # setter of suppport range
    def set_ruv_range(self, rmin, rmax, umin):
        self.rmin = rmin
        self.rmax = rmax
        self.umin = umin
        self.umax = 1.0
        self.vmin = 0.0
        self.vmax = 1.0

    def set_ellpsimu_range(self, ellmin, ellmax, psimin, epmu=1e-7):
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.psimin = psimin
        self.psimax = np.pi/4
        self.mumin  = -1.0 + epmu
        self.mumax  = 1.0

        # When ell, psi, mu run over the lectangular region
        # defined by the above ranges, the ranges of r, u, v
        # are also given by the ranges on ell, psi, mu.
        rmin = ellmin*min(5**-0.5, (1-(1-epmu)*2/5)**0.5)
        rmax = ellmax*max(2**-0.5, np.cos(psimin))
        umin = min(2**0.5*epmu**0.5, np.tan(psimin))
        self.set_ruv_range(rmin, rmax, umin)

    def set_scale_range(self, ellmin, ellmax):
        ellmin = 2**0.5*ell12min
        ellmax = 2**0.5*ell12max
        psimin = min(np.arctan2(ell12min, ell12max), np.pi/2 - np.arctan(ell12max/ell12min))


    def matter_bispectrum(self, k1, k2, k3, z):
        raise NotImplementedError

    # kappa bispectrum interface
    def kappa_bispectrum(self, ell1, ell2, ell3, method='direct', **args):
        """
        Compute kappa bispectrum.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        method (str): method for computing kappa bispectrum
        """
        if method == 'direct':
            return self.kappa_bispectrum_direct(ell1, ell2, ell3, **args)
        elif method == 'interp':
            return self.kappa_bispectrum_interp(ell1, ell2, ell3)
        elif method == 'resum':
            return self.kappa_bispectrum_resum(ell1, ell2, ell3)
        else:
            raise ValueError("method must be 'direct', 'interp', or 'resum'")
        
    # direct evaluation of kappa bispectrum from matter bispectrum
    def kappa_bispectrum_direct(self, ell1, ell2, ell3, zmin=1e-2, nzbin=20, **args):
        """
        Compute kappa bispectrum by direct line-of-sight integration.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        zmin (float): minimum redshift
        nzbin (int): number of redshift bins
        args (dict): arguments for matter_bispectrum
        """
        # check scalar
        isscalar = np.isscalar(ell1)
        if isscalar:
            ell1 = np.array([ell1])
            ell2 = np.array([ell2])
            ell3 = np.array([ell3])

        # check shape
        if ell1.shape != ell2.shape or ell1.shape != ell3.shape:
            raise ValueError("l1, l2, l3 must have the same shape")

        # save input shape
        shape = ell1.shape

        # reshape to 1d
        ell1 = ell1.ravel()
        ell2 = ell2.ravel()
        ell3 = ell3.ravel()

        # compute lensing weight, encoding geometrical dependence.
        z = np.logspace(np.log10(zmin), np.log10(self.zmax), nzbin)
        chi = self.z2chi(z)
        g   = self.chi2g(chi)
        weight = g**3/chi*(1+z)**3

        # create grids
        ELL1, Z = np.meshgrid(ell1, z, indexing='ij')
        ELL2, Z = np.meshgrid(ell2, z, indexing='ij')
        ELL3, Z = np.meshgrid(ell3, z, indexing='ij')
        CHI = self.z2chi(Z)
        K1, K2, K3 = ELL1/CHI, ELL2/CHI, ELL3/CHI

        # integrand
        i = weight * self.matter_bispectrum(K1, K2, K3, Z, **args)

        # integrate
        bk = np.trapz(i, chi, axis=1)

        # Multiply prefactor
        bk *= (3/2 * (100/299792)**2 * self.cosmo.Om0)**3

        # reshape to the original shape
        bk = bk.reshape(shape)

        # convert to scalar if input is scalar
        if isscalar:
            bk = bk[0]

        return bk

    # interpolation
    def interpolate(self, nrbin=20, nubin=25, nvbin=20, method='linear', nzbin=20, **args):
        """
        Interpolate kappa bispectrum.

        nrbin (int): number of r bins
        nubin (int): number of u bins
        nvbin (int): number of v bins
        method (str): method for interpolation
        nzbin (int): number of redshift bins
        args (dict): arguments for matter_bispectrum
        """
        r = np.logspace(np.log10(self.rmin), np.log10(self.rmax), nrbin)
        u = np.logspace(np.log10(self.umin), np.log10(self.umax), nubin)
        v = np.linspace(self.vmin, self.vmax, nvbin)

        R, U, V = np.meshgrid(r, u, v, indexing='ij')
        ell1, ell2, ell3 = trigutils.ruv_to_x1x2x3(R, U, V)
        bk = self.kappa_bispectrum_direct(ell1, ell2, ell3, nzbin=nzbin, **args)

        self.ip = rgi((np.log(r), np.log(u), v), np.log(bk), method=method)

    def kappa_bispectrum_interp(self, ell1, ell2, ell3):
        """
        Compute kappa bispectrum by interpolation.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        """
        r, u, v = trigutils.x1x2x3_to_ruv(ell1, ell2, ell3, signed=False)
        x = edge_correction(np.log(r), self.ip.grid[0].min(), self.ip.grid[0].max())
        y = edge_correction(np.log(u), self.ip.grid[1].min(), self.ip.grid[1].max())
        z = edge_correction(v, self.ip.grid[2].min(), self.ip.grid[2].max()) 
        return np.exp(self.ip((x,y,z)))

    # multipole decomposition
    def init_multipole(self, Lmax, MU, method='linear'):
        if (not hasattr(self, 'multipole')) or self.multipole.Lmax != Lmax or self.multipole.x.shape != MU.shape or np.any(self.multipole.x != MU) or self.multipole.method != method:
            self.multipole = Multipole(MU, Lmax, method=method, verbose=True)

    def decompose(self, Lmax, nellbin=100, npsibin=100, nmubin=100, 
            method_decomp='linear', method_bispec='interp'):
        """
        Compute multipole decomposition of kappa bispectrum.

        Lmax (int): maximum multipole
        nellbin (int): number of ell bins
        npsibin (int): number of psi bins
        nmubin (int): number of mu bins
        method_decomp (str): method for multipole decomposition
        method_bispec (str): method for computing bispectrum

        Note that the bispectrum is specified by two sides and its *outer* angle,
        while the multipole decomposition is defined with the *inner* angle.
        """
        ell = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), nellbin)
        psi = loglinear(self.psimin, 1e-3, 1e-2, self.psimax, npsibin//10, npsibin - npsibin//10)
        mu  = np.linspace(self.mumin, self.mumax, nmubin)
        ELL, SPI, MU = np.meshgrid(ell, psi, mu, indexing='ij')
        self.init_multipole(Lmax, MU, method_decomp)

        # Note that the bispectrum is specified by two sides and its inner angle,
        # while the multipole decomposition is defined with the outer angle.
        # Thus, we need a minus sign for mu.
        ELL1, ELL2, ELL3 = trigutils.xpsimu_to_x1x2x3(ELL, SPI, -MU)
        b = self.kappa_bispectrum(ELL1, ELL2, ELL3, method=method_bispec)

        # Compute multipoles
        L = np.arange(Lmax+1)
        bL = self.multipole.decompose(b, L, axis=2)
        self.multipoles_data = {'ell':ell, 'psi':psi, 'bL':bL}

    def _kappa_bispectrum_multipole(self, L, ell, psi):
        """
        Compute multipole of kappa bispectrum.

        L (int): multipole
        ell (array): ell array
        psi (array): psi array
        """
        Lmax = self.multipoles_data['bL'].shape[0]
        if L > Lmax:
            raise ValueError("L must be less than Lmax={}".format(Lmax))

        x = np.log(self.multipoles_data['ell'])
        y = np.log(self.multipoles_data['psi'])
        z = self.multipoles_data['bL'][L, :, :]
        f = rgi((x, y), z, bounds_error=True)

        # convert psi to pi/2-psi if psi > pi/4
        psi = psi.copy()
        sel = np.pi/4 < psi
        psi[sel] = np.pi/2 - psi[sel]

        x = edge_correction(np.log(ell), f.grid[0].min(), f.grid[0].max())
        y = edge_correction(np.log(psi), f.grid[1].min(), f.grid[1].max())
        out = f((x, y))

        return out

    def kappa_bispectrum_multipole(self, L, ell, psi):
        """
        Compute multipole of kappa bispectrum.

        L (int or array): multipole
        ell (array): ell array
        psi (array): psi array
        """
        if np.isscalar(L):
            L = np.array([L])
        out = np.zeros((L.size,) + ell.shape)
        for i, _L in enumerate(L):
            out[i] = self._kappa_bispectrum_multipole(_L, ell, psi)
        return out

    def kappa_bispectrum_resum(self, ell1, ell2, ell3):
        """
        Compute kappa bispectrum by resummation of multipoles.

        ell1 (array): ell1 array
        ell2 (array): ell2 array
        ell3 (array): ell3 array
        """
        ell, psi, mu = trigutils.x1x2x3_to_xpsimu(ell1, ell2, ell3)
        Lmax = self.multipoles_data['bL'].shape[0]
        L = np.arange(Lmax)
        bL = self.kappa_bispectrum_multipole(L, ell, psi)
        pL = np.array([eval_legendre(_L, -mu) for _L in L])
        out = np.sum(bL*pL, axis=0)
        return out


class BispectrumHalofit(BispectrumBase):
    def __init__(self, cosmo=None, zs=None, pzs=None):
        self.halofit = Halofit()
        super().__init__(cosmo, zs, pzs)

    def set_cosmology(self, cosmo):
        """
        """
        super().set_cosmology(cosmo)
        # parameters for halofit
        dcosmo={'Om0': cosmo.Om0, 
                'Ode0': cosmo.Ode0,
                'ns': cosmo.meta.get('n'),
                'sigma8': cosmo.meta.get('sigma8'), 
                'w0': cosmo.w0, 
                'wa': 0.0,
                'fnu0': 0.0} 
        self.halofit.set_cosmology(dcosmo)

    def set_pklin(self, k, pklin):
        self.halofit.set_pklin(k, pklin)

    def set_lgr(self, z, lgr):
        self.halofit.set_lgr(z, lgr)

    def matter_bispectrum(self, k1, k2, k3, z, all_physical=True, which=['Bh1', 'Bh3']):
        return self.halofit.get_bihalofit(k1, k2, k3, z, all_physical=all_physical, which=which)

class BispectrumNFW1Halo(BispectrumBase):
    """
    Toy model
    """
    def __init__(self, cosmo=None, zs=None, pzs=None, rs=10.0):
        super().__init__(cosmo, zs, pzs)
        self.set_rs(rs)

    def set_rs(self, rs):
        self.rs = rs

    @classmethod
    def rhok_NFW(cls, k, rs):
        y = k*rs
        si, ci = sici(y)
        return -np.cos(y)*ci + 0.5*np.sin(y)*(np.pi-2*si)

    def kappa_bispectrum_direct(self, ell1, ell2, ell3, **args):
        rs = np.deg2rad(self.rs/60.0) # in rad
        bl = 1
        for i, _ell in enumerate([ell1, ell2, ell3]):
            bl *= self.rhok_NFW(_ell, rs)
        return bl