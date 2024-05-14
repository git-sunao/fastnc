"""
Author: Sunao Sugiyama
Last edit: 2024/01/18
"""
import numpy as np
from . import fftlog
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from . import trigutils
from .utils import loglinear

class OneJNaturalConponent:
    """
    See https://arxiv.org/abs/astro-ph/0308328 and https://arxiv.org/abs/2208.11686
    """
    def __init__(self, bispectrum=None):
        if bispectrum is not None:
            self.set_bispectrum(bispectrum)

    def set_bispectrum(self, bispectrum, method='interp'):
        self.bispectrum = bispectrum
        self.method_bispec = method

        self.ellmin = self.bispectrum.ellmin
        self.ellmax = self.bispectrum.ellmax
        self.psimin = self.bispectrum.psimin
        self.psimax = self.bispectrum.psimax
        self.mumin  = self.bispectrum.mumin
        self.mumax  = self.bispectrum.mumax

    def expibarbeta(self, psi, dbeta):
        cos2b = np.cos(dbeta) + np.sin(2*psi)
        sin2b = np.cos(2*psi) * np.sin(dbeta)
        norm  = np.sqrt(cos2b**2 + sin2b**2)
        out = cos2b/norm + 1j*sin2b/norm
        return out**0.5
    
    def expialpha(self, psi, dbeta, tau, dvphi):
        zeta = 0.5*(dbeta-dvphi)
        cosa =  np.cos(psi+tau)*np.sin(zeta)
        sina = -np.cos(psi-tau)*np.cos(zeta)
        norm = np.sqrt(cosa**2 + sina**2)
        eia = cosa/norm + 1j*sina/norm
        return eia

    def Atilde(self, psi, dbeta, tau, dvphi):
        zeta = 0.5*(dbeta-dvphi)
        out = np.cos(psi-tau)**2*np.cos(zeta)**2 + np.cos(psi+tau)**2*np.sin(zeta)**2
        out = np.sqrt(out)
        return out

    def get_f_of_logA(self, psi, dbeta, order=6, extrap=False):
        ell = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), 256)
        ell1, ell2, ell3 = trigutils.xpsimu_to_x1x2x3(ell, psi, -np.cos(dbeta))
        bs = self.bispectrum.kappa_bispectrum(ell1, ell2, ell3, method=self.method_bispec)
        # extrap for FFTLog
        N_extrap_high = int(0.1*ell.size) if bs[-1] != 0 and extrap else 0
        N_extrap_low  = int(0.1*ell.size) if bs[0] != 0 and extrap else 0
        hankel = fftlog.hankel(ell, ell**4*bs, N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high, N_pad=200)
        A, f = hankel.hankel(order)
        f_of_logA = ius(np.log(A), A*f, ext=1)
        return f_of_logA

    def integrand_sym(self, t, tau, dvphi, psi, dbeta):
        f6_of_logA = self.get_f_of_logA(psi, dbeta, order=6)
        f2_of_logA = self.get_f_of_logA(psi, dbeta, order=2)

        o0 = 0
        o1 = 0
        o2 = 0
        o3 = 0
        for _psi in [psi, np.pi/2-psi]:
            for _dbeta in [dbeta, -dbeta]:
                Atil = self.Atilde(_psi, _dbeta, tau, dvphi)
                f6 = f6_of_logA(np.log(t*Atil))/(t*Atil)
                f2 = f2_of_logA(np.log(t*Atil))/(t*Atil)

                eb = self.expibarbeta(_psi, _dbeta)
                ea = self.expialpha(_psi, _dbeta, tau, dvphi)
                ez = np.exp(0.5j*(_dbeta-dvphi))
                si = np.sin(2*_psi)

                o0+= si * eb**2 * ea**-6 * f6
                o1+= si * eb**-2* ea**-2 * f2
                o2+= si * eb**2 * ea**-2 * f2 * ez**-4
                o3+= si * eb**2 * ea**-2 * f2 * ez**4

        return o0, o1, o2, o3

    def compute(self, t, tau, phi, projection='x', nbin_psi=100, nbin_dbeta=101):
        """
        t (array)
        tau (float)
        phi (float)
        """
        # psi = loglinear(self.psimin, 1e-3, self.psimax, 60, 60)
        psi = np.linspace(self.psimin, self.psimax, nbin_psi)
        # dbeta = np.pi-np.arccos(1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, 70, 60)[::-1])
        dbeta = np.pi-np.arccos(1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, nbin_dbeta, nbin_dbeta)[::-1])

        dvphi = -phi

        gam0 = []
        gam1 = []
        gam2 = []
        gam3 = []
        for _psi in psi:
            _0 = []
            _1 = []
            _2 = []
            _3 = []
            for _dbeta in dbeta:
                o0, o1, o2, o3 = self.integrand_sym(t, tau, dvphi, _psi, _dbeta)
                gam0.append(o0)
                gam1.append(o1)
                gam2.append(o2)
                gam3.append(o3)
        gam0 = np.reshape(gam0, (len(psi), len(dbeta), -1))
        gam1 = np.reshape(gam1, (len(psi), len(dbeta), -1))
        gam2 = np.reshape(gam2, (len(psi), len(dbeta), -1))
        gam3 = np.reshape(gam3, (len(psi), len(dbeta), -1))
        gam0 = np.trapz(gam0, psi, axis=0)
        gam1 = np.trapz(gam1, psi, axis=0)
        gam2 = np.trapz(gam2, psi, axis=0)
        gam3 = np.trapz(gam3, psi, axis=0)
        gam0 = np.trapz(gam0, dbeta, axis=0)/2/(2*np.pi)**3
        gam1 = np.trapz(gam1, dbeta, axis=0)/2/(2*np.pi)**3
        gam2 = np.trapz(gam2, dbeta, axis=0)/2/(2*np.pi)**3
        gam3 = np.trapz(gam3, dbeta, axis=0)/2/(2*np.pi)**3

        self.gam0 = gam0
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3