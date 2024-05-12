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

    def exp2ibarbeta(self, psi, dbeta):
        cos2b = np.cos(dbeta) + np.sin(2*psi)
        sin2b = np.cos(2*psi) * np.sin(dbeta)
        norm  = np.sqrt(cos2b**2 + sin2b**2)
        out = cos2b/norm + 1j*sin2b/norm
        return out
    
    def expiNalpha(self, psi, dbeta, tau, phi, N=6):
        cosa = np.cos(tau-psi) * np.cos((dbeta+phi)/2)
        sina = np.cos(tau+psi) * np.sin((dbeta+phi)/2)
        norm = np.sqrt(cosa**2 + sina**2)
        eia = cosa/norm - 1j*sina/norm
        out = eia**N
        return out

    def r(self, psi, dbeta, tau, phi):
        out = (np.cos(tau)*np.cos(psi))**2 + (np.sin(tau)*np.sin(psi))**2 + 0.5*np.sin(2*tau)*np.sin(2*psi)*np.cos(dbeta+phi)
        out = np.sqrt(out)
        return out

    def get_f_of_logA(self, psi, dbeta, order=6, extrap=False):
        ell = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), 1024)
        ell1, ell2, ell3 = trigutils.xpsimu_to_x1x2x3(ell, psi, -np.cos(dbeta))
        bs = self.bispectrum.kappa_bispectrum(ell1, ell2, ell3, method=self.method_bispec)
        # extrap for FFTLog
        N_extrap_high = int(0.1*ell.size) if bs[-1] != 0 and extrap else 0
        N_extrap_low  = int(0.1*ell.size) if bs[0] != 0 and extrap else 0
        hankel = fftlog.hankel(ell, ell**4*bs, N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high, N_pad=400)
        A, f = hankel.hankel(order)
        f_of_logA = ius(np.log(A), A*f, ext=1)
        return f_of_logA

    def integrand0(self, t, tau, phi, psi, dbeta):
        f_of_logA = self.get_f_of_logA(psi, dbeta, order=6)

        out = 0
        for _psi in [psi, np.pi/2-psi]:
            for _dbeta in [dbeta, -dbeta]:
                r = self.r(_psi, _dbeta, tau, phi)
                p = self.expiNalpha(_psi, _dbeta, tau, phi, N=6)
                p2= np.sin(2*_psi)*self.exp2ibarbeta(_psi, _dbeta)
                A = t*r
                out+= f_of_logA(np.log(A))*p*p2/A

        return out
    
    def integrand1(self, t, tau, phi, psi, dbeta):
        f_of_logA = self.get_f_of_logA(psi, dbeta, order=2)

        out = 0
        for _psi in [psi, np.pi/2-psi]:
            for _dbeta in [dbeta, -dbeta]:
                r = self.r(_psi, _dbeta, tau, phi)
                p = self.expiNalpha(_psi, _dbeta, tau, phi, N=2)
                p2= np.sin(2*_psi)*self.exp2ibarbeta(_psi, _dbeta) * np.exp(-2j*_dbeta)
                A = t*r
                out+= f_of_logA(np.log(A))*p*p2/A

        return out
    
    def integrand2(self, t, tau, phi, psi, dbeta):
        f_of_logA = self.get_f_of_logA(psi, dbeta, order=2)

        out = 0
        for _psi in [psi, np.pi/2-psi]:
            for _dbeta in [dbeta, -dbeta]:
                r = self.r(_psi, _dbeta, tau, phi)
                p = self.expiNalpha(_psi, _dbeta, tau, phi, N=2)
                p2= np.sin(2*_psi)*self.exp2ibarbeta(_psi, _dbeta) * np.exp(+2j*_dbeta) * np.exp(2j*phi)
                A = t*r
                out+= f_of_logA(np.log(A))*p*p2/A

        return out

    def integrand3(self, t, tau, phi, psi, dbeta):
        f_of_logA = self.get_f_of_logA(psi, dbeta, order=2)

        out = 0
        for _psi in [psi, np.pi/2-psi]:
            for _dbeta in [dbeta, -dbeta]:
                r = self.r(_psi, _dbeta, tau, phi)
                p = self.expiNalpha(_psi, _dbeta, tau, phi, N=2)
                p2= np.sin(2*_psi)*np.conj(self.exp2ibarbeta(_psi, _dbeta)) * np.exp(-2j*phi)
                A = t*r
                out+= f_of_logA(np.log(A))*p*p2/A

        return out

    def GammaN(self, t, tau, phi, N, projection='x', nbin_psi=100, nbin_dbeta=100):
        """
        t (array)
        tau (float)
        phi (float)
        """
        # psi = loglinear(self.psimin, 1e-3, self.psimax, 60, 60)
        psi = np.linspace(self.psimin, self.psimax, nbin_psi)
        # dbeta = np.pi-np.arccos(1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, 70, 60)[::-1])
        dbeta = np.pi-np.arccos(1-loglinear(1-self.mumax, 5e-2, 1-self.mumin, nbin_dbeta, nbin_dbeta)[::-1])

        out = []
        for _psi in psi:
            _ = []
            for _dbeta in dbeta:
                if N == 0:
                    v = self.integrand0(t, tau, phi, _psi, _dbeta)
                elif N == 1:
                    v = self.integrand1(t, tau, phi, _psi, _dbeta)
                elif N == 2:
                    v = self.integrand2(t, tau, phi, _psi, _dbeta)
                elif N == 3:
                    v = self.integrand3(t, tau, phi, _psi, _dbeta)
                _.append(v)
            out.append(np.trapz(_, dbeta, axis=0))
        out = -1/(2*np.pi)**3/2 * np.trapz(out, psi, axis=0)

        return out

    def Gamma0(self, t, tau, phi, projection='x', nbin_psi=100, nbin_dbeta=100):
        return self.GammaN(t, tau, phi, 0, projection=projection, nbin_psi=nbin_psi, nbin_dbeta=nbin_dbeta)
    
    def Gamma1(self, t, tau, phi, projection='x', nbin_psi=100, nbin_dbeta=100):
        return self.GammaN(t, tau, phi, 1, projection=projection, nbin_psi=nbin_psi, nbin_dbeta=nbin_dbeta)
    
    def Gamma2(self, t, tau, phi, projection='x', nbin_psi=100, nbin_dbeta=100):
        return self.GammaN(t, tau, phi, 2, projection=projection, nbin_psi=nbin_psi, nbin_dbeta=nbin_dbeta)

    def Gamma3(self, t, tau, phi, projection='x', nbin_psi=100, nbin_dbeta=100):
        return self.GammaN(t, tau, phi, 3, projection=projection, nbin_psi=nbin_psi, nbin_dbeta=nbin_dbeta)