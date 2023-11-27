"""
This is an implementation of natural component of 3PCF.
This is based on Eq. (22) of Heyndenreich+2022.
In this code , we utilize FFTLog to perfom R integral convolved with Bessel function.
For now the accuracy is *not* tested so much. Just for rough comparison to other method.

Author: Sunao Sugiyama
Last edit: 2023/11/06
"""
import numpy as np
from . import fftlog
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from . import trigutils

class SHCalculator:
    def __init__(self, bispectrum, lmin, lmax):
        self.bispectrum = bispectrum
        self.lmin = lmin
        self.lmax = lmax

    def sincos2angbar(self, psi, delta):
        cos2b = np.cos(delta) + np.sin(2*psi)
        sin2b = np.cos(2*psi) * np.sin(delta)
        norm  = np.sqrt(cos2b**2 + sin2b**2)
        return sin2b/norm, cos2b/norm

    def bJ6(self, psi, varphi):
        R = np.logspace(np.log10(self.lmin), np.log10(self.lmax), 1024)
        b = self.bispectrum(R, psi, -np.cos(varphi))
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
        sin2bb, cos2bb = self.sincos2angbar(psi, varphi)
        expi6a = self.expi6alpha(x1, x2, phi3, psi, varphi)
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

    def Gamma0(self, x1, x2, phi3, Nvarphi=50, Npsi=51, pad=1e-4):
        varphi = np.linspace(pad, np.pi-pad, Nvarphi)
        psi    = np.linspace(pad, np.pi/4, Npsi)
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

        return out

    def Gamma0_treecorr(self, r, u, v, Nvarphi=50, Npsi=51, pad=1e-4, tid=0):
        if tid==0:
            x1, x2, dvarphi = trigutils.ruv_to_x1x2dvphi(r, u, v)
            if v>0:
                phi3 = np.pi+dvarphi
            else:
                phi3 = np.pi-dvarphi
        elif tid==1:
            x1, x2, dvarphi = trigutils.ruv_to_x2x3dvphi(r, u, v)
            if v>0:
                phi3 = np.pi+dvarphi
            else:
                phi3 = np.pi-dvarphi
        elif tid==2:
            x1, x2, dvarphi = trigutils.ruv_to_x3x1dvphi(r, u, v)
            if v>0:
                phi3 = np.pi+dvarphi
            else:
                phi3 = np.pi-dvarphi

        return self.Gamma0(x1, x2, phi3, Nvarphi=Nvarphi, Npsi=Npsi, pad=pad)