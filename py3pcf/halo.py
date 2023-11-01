import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from . import fftlog
from scipy.special import sici

class halo_baseclass:
    def __init__(self, name=''):
        self.name = name
        
    def set_param(self, params):
        self.params = params
        
    def rbins(self, bins=1024):
        """
        This method should give appropriate r bins, 
        which covers the typical scale of the halo profile.
        """
        NotImplemented
        
    def kbins(self, r_req=None, k_req=None, bins=1024):
        return 1.0/self.rbins(r_req=r_req, k_req=k_req, bins=bins)[::-1]
        
    def rho_of_r(self, r):
        NotImplemented
        
    def rho_of_k(self, k, Nel=1024, Neh=1024):
        r   = self.rbins(k_req=k)
        rho = self.rho_of_r(r)
        flg = fftlog.fftlog(r, 4*np.pi*r**3*rho, \
                            N_extrap_low=Nel, N_extrap_high=Neh)
        out = flg.fftlog(0)
        rhok= ius(out[0], out[1])(k)
        return rhok
    
    def rho_of_k_analytic(self, k):
        NotImplemented
        
    def Sigma_of_R(self, R, Nel=1024, Neh=1024):
        """
        Compute the surface mass density.
        This can be obtained by integrating the mass 
        profile along the line of sight. 
        
        .. math:: 
            \\Sigma(R) = \\int {\\rm d}z~\\rho\\left(\\sqrt{R^2 + z^2}\\right)
        
        This can be easily calcurated from the Fourier counter part.
        
        .. math::
            \\Sigma(R) = \\int \\frac{k{\\rm k}}{(2\\pi) \\rho(k) J_0(kR)}
        
        where :math:`\\rho(k)` is the 3 dimensional 
        Fourier transformation of :math:`\\rho(r)`.
        """
        k   = self.kbins(r_req=R)
        rhok= self.rho_of_k(k)
        hkl = fftlog.hankel(k, k**2*rhok/2/np.pi, \
                            N_extrap_low=Nel, N_extrap_high=Neh)
        out = hkl.hankel(0)
        SigR = ius(out[0], out[1])(R)
        return SigR
    
    def DeltaSigma_of_R(self, R, Nel=1024, Neh=2014):
        """
        Compute the surface mass density.
        This can be obtained by integrating the mass 
        profile along the line of sight. 
        
        .. math:: 
            \\Delta\\Sigma(R) = \\bar{\\Sigma}(R) - \\Sigma(R)
        
        This can be easily calcurated from the Fourier counter part.
        
        .. math::
            \\Delta\\Sigma(R) = \\int \\frac{k{\\rm k}}{(2\\pi) \\rho(k) J_2(kR)}
        
        where :math:`\\rho(k)` is the 3 dimensional 
        Fourier transformation of :math:`\\rho(r)`.
        """
        k   = self.kbins(r_req=R)
        rhok= self.rho_of_k(k)
        hkl = fftlog.hankel(k, k**2*rhok/2/np.pi, \
                            N_extrap_low=Nel, N_extrap_high=Neh)
        out = hkl.hankel(2)
        SigR = ius(out[0], out[1])(R)
        return SigR
    
    def kappa_of_theta(self, theta, Sigmacrit, dA):
        R = theta*dA
        k = self.Sigma_of_R(R)/Sigmacrit
        return k
    
    def Deltakappa_of_theta(self, theta, Sigmacrit, dA):
        R = theta*dA
        dk= self.DeltaSigma_of_R(R)/Sigmacrit
        return dk
        
    def gamma_of_theta(self, theta1, theta2, Sigmacrit, dA):
        """
        Return complex shear, defined with the 
        shear reference point at halo center.
        """
        theta = np.sqrt(theta1**2+theta2**2)
        dk    = self.Deltakappa_of_theta(theta, Sigmacrit, dA)
        phi   = np.arctan(theta2, theta1)
        phase = np.exp(2j*phi)
        return dk * phase
    
    def kappa_of_l(self, l, dA):
        return self.rho_of_k(l/dA)
    
class NFW(halo_baseclass):
    def __init__(self, name='NFW'):
        super().__init__(name)
        
    def rbins(self, r_req=None, k_req=None, bins=1024):
        """
        Gives appropriate r bins, 
        which covers the typical scale of the halo profile.
        """
        # typical scale
        rs    = self.params['rs']
        logrs = np.log10(rs)
        if r_req is None and k_req is None:
            logrmin, logrmax = logrs-3, logrs+3
        elif r_req is None:
            logrmin, logrmax =-np.log10(k_req.max()),-np.log10(k_req.min())
        elif k_req is None:
            logrmin, logrmax = np.log10(r_req.min()), np.log10(r_req.max())
        else:
            logrmin = min(-np.log10(k_req.max()), np.log10(r_req.min()))
            logrmax = max(-np.log10(k_req.min()), np.log10(r_req.max()))
        
        # symmetrize the range with respect to the typical scale
        side = max(abs(logrs-logrmin), abs(logrmax-logrs))
        logrmin, logrmax = logrs-side, logrs+side

        # if the range is too narrow, we go back to the default choice
        if (logrs-logrmin)<3 or (logrmax-logrs)<3:
            logrmin, logrmax = logrs-3, logrs+3
        
        r = np.logspace(logrmin, logrmax, bins)
        return r
        
    def rho_of_k_analytic(self, k):
        rs  = self.params['rs']
        rhos= self.params['rhos']
        pref= 4*np.pi * rhos*rs**3
        y   = k*rs
        si, ci = sici(y)
        return  pref*(-np.cos(y)*ci + 0.5*np.sin(y)*(np.pi-2*si))

    def rho_of_r(self, r):
        # get params
        rs  = self.params['rs']
        rhos= self.params['rhos']
        
        # profile
        x = r/rs
        p = rhos/x/(1+x)**2
        
        return p