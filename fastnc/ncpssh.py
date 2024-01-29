"""
Author: Sunao Sugiyama
Last edit: 2024/01/18
"""
import numpy as np
from . import fftlog
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from . import trigutils

class NaturalConponentPSSH:
    """
    See https://arxiv.org/abs/astro-ph/0308328 and https://arxiv.org/abs/2208.11686
    """
    def __init__(self, ellmin, ellmax, psimin, psimax, mumin, mumax):
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.psimin = psimin
        self.psimax = psimax
        self.mumin  = mumin
        self.mumax  = mumax

    def set_bispectrum(self, bispectrum, method='interp'):
        self.bispectrum = bispectrum
        self.method_bispec = method

    def sincos2angbar(self, psi, delta):
        cos2b = np.cos(delta) + np.sin(2*psi)
        sin2b = np.cos(2*psi) * np.sin(delta)
        norm  = np.sqrt(cos2b**2 + sin2b**2)
        return sin2b/norm, cos2b/norm

    def bJ6(self, psi, varphi):
        """
        Performs R integral in Eq. (19) of https://arxiv.org/abs/2208.11686
        """
        R = np.logspace(np.log10(self.ellmin), np.log10(self.ellmax), 1024)
        ell1, ell2, ell3 = trigutils.xpsimu_to_x1x2x3(R, psi, -np.cos(varphi))
        b = self.bispectrum.kappa_bispectrum(ell1, ell2, ell3, method=self.method_bispec)
        f = R**4*b
        if f[-1] != 0:
            N_extrap_high = int(0.1*R.size)
        else:
            N_extrap_high = 0
        if f[0] != 0:
            N_extrap_low = int(0.1*R.size)
        else:
            N_extrap_low = 0
        h = fftlog.hankel(R, f, N_extrap_low=N_extrap_low, N_extrap_high=N_extrap_high)
        A, f = h.hankel(6)
        return A, f

    def expi6alpha(self, x1, x2, phi3, psi, varphi):
        c = (x2*np.cos(psi) + x1*np.sin(psi)) * np.cos((varphi+phi3)/2)
        s = (x2*np.cos(psi) - x1*np.sin(psi)) * np.sin((varphi+phi3)/2)
        n = np.sqrt(c**2 + s**2)
        eia = c/n - 1j*s/n
        ei6a= eia**6
        return ei6a

    def A(self, x1, x2, phi3, psi, varphi):
        A = np.sqrt( (x2*np.cos(psi))**2 + (x1*np.sin(psi))**2 + x1*x2*np.sin(2*psi)*np.cos(varphi+phi3))
        return A

    def integrand(self, f_interp, x1, x2, phi3, psi, varphi):
        # exp 2i bar{beta}
        sin2bb, cos2bb = self.sincos2angbar(psi, varphi)
        # exp 6i alpha in Eq. (21)
        expi6a = self.expi6alpha(x1, x2, phi3, psi, varphi)
        # A in Eq. (20)
        A = self.A(x1, x2, phi3, psi, varphi)
        f = f_interp(np.log(A))
        o = 0.5*np.sin(2*psi) * (cos2bb + 1j*sin2bb) * expi6a * f
        return o

    def integrand_sym(self, x1, x2, phi3, psi, varphi):
        A_base, f_base = self.bJ6(psi, varphi)
        f_interp = ius(np.log(A_base), f_base, ext='zeros')

        out = 0
        # 0,0
        out+= self.integrand(f_interp, x1, x2, phi3, psi, varphi)
        # 1,0
        out+= self.integrand(f_interp, x1, x2, phi3, psi, 2*np.pi-varphi)
        # 0,1
        out+= self.integrand(f_interp, x1, x2, phi3, np.pi/2-psi, varphi)
        # 1,1
        out+= self.integrand(f_interp, x1, x2, phi3, np.pi/2-psi, 2*np.pi-varphi)
        return out

    def _phase_orthocenter2centroid(self, i, x1, x2, dvarphi):
        #https://arxiv.org/pdf/astro-ph/0207454.pdf between eq 12 and 13
        x1, x2, x3 = trigutils.x1x2dvphi_to_x1x2x3(x1, x2, dvarphi)

        def temp(x1, x2, x3):
            phi3 = np.arccos( (x1**2+x2**2-x3**2)/2/x1/x2 )
            cos2psi = ((x2**2-x1**2)**2 - 4*x1**2*x2**2*np.sin(phi3)**2)/4.0
            sin2psi = (x2**2-x1**2) * x1*x2 * np.sin(phi3)
            norm = np.sqrt(cos2psi**2 + sin2psi**2)
            exp2psi = cos2psi/norm + 1j*sin2psi/norm
            return exp2psi

        exp2psi3 = temp(x1, x2, x3)
        exp2psi1 = temp(x2, x3, x1)
        exp2psi2 = temp(x3, x1, x2)

        out = 1
        for j, phase in enumerate([1.0, exp2psi1, exp2psi2, exp2psi3]):
            if j == i:
                out *= phase
            else:
                out *= np.conj(phase)

        return out

    def phase(self, i,x1,x2,dvphi,projection):
        if projection=='cent':
            return self._phase_orthocenter2centroid(i,x1,x2,dvphi)
        else:
            raise NotImplementedError

    def Gamma0(self, x1, x2, phi3, npsibin=50, nvarphibin=50, projection='cent'):
        psi    = np.linspace(self.psimin, self.psimax, npsibin)
        varphi = np.linspace(np.arccos(self.mumax), np.arccos(self.mumin), nvarphibin)

        dvarphi= varphi[1]-varphi[0]
        dpsi   = psi[1]-psi[0]

        out = []
        for _varphi in varphi:
            sub = []
            for _psi in psi:
                i = self.integrand_sym(x1, x2, phi3, _psi, _varphi)
                sub.append(-i/(2*np.pi)**3)
            out.append( np.trapz(sub, psi, axis=0) )
        out = np.trapz(out, varphi, axis=0)

        x3 = np.sqrt(x1**2+x2**2-2*x1*x2*np.cos(phi3))
        phi1 = np.arccos( (x2**2+x3**2-x1**2)/2/x2/x3)
        phi2 = np.arccos( (x1**2+x3**2-x2**2)/2/x1/x3)

        out *= np.exp(1j*(phi1-phi2))

        out *= self.phase(0, x1, x2, phi3-np.pi, projection)

        out *= -1

        return out

    def Gamma0_treecorr(self, r, u, v, npsibin=50, nvarphibin=50, projection='cent'):
        x1, x2, dvarphi = trigutils.ruv_to_x1x2dvphi(r, u, v)
        phi3 = np.pi+dvarphi
        return self.Gamma0(x1, x2, phi3, npsibin=npsibin, nvarphibin=nvarphibin, projection=projection)