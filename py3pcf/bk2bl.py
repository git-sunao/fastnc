import numpy as np
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator as rgi
from . import utils

class bk2bl:
    def __init__(self, cosmo, bispectrum, **kwargs):
        self.cosmo      = cosmo
        self.bispectrum = bispectrum
        self.kwargs     = kwargs

    def set_kernel(self, zs):
        chis = self.cosmo.z2chi(zs)
        chi = np.logspace(-1, np.log10(chis), 50)
        z   = self.cosmo.chi2z(chi)
        H0  = self.cosmo.H0.value/self.cosmo.h # 1/Mpc
        Om  = self.cosmo.Om0
        g   = 3.0/2.0 * H0**2*Om * (chis-chi)/chis

        self.chi   = chi
        self.g_dist= g

    def bl_direct(self, targ1, targ2, targ3, targtype='l1l2l3'):
        # Get l1, l2, l3
        l1, l2, l3 = utils.get_l1l2l3(targ1, targ2, targ3, targtype)

        # make grid for chi, l, and z
        _chi, _l1 = np.meshgrid(self.chi, l1.reshape(-1))
        _chi, _l2 = np.meshgrid(self.chi, l2.reshape(-1))
        _chi, _l3 = np.meshgrid(self.chi, l3.reshape(-1))
        _z = self.cosmo.chi2z(_chi)

        # Get matter power spectrum
        bk = self.bispectrum(_l1/_chi, _l2/_chi, _l3/_chi, _z, **self.kwargs)

        # Perform line-of-sight integration
        bl = simps((1+_z)**3*self.g_dist**3/_chi*bk, _chi)

        # reshape to the original l shape
        bl = bl.reshape(l1.shape)

        return bl

    def prepare_interpolation(self, Nl=50, Npsi=30, Nu=40, domain='domain1'):
        self.domain = domain
        if self.domain == 'domain1':
            l  = np.logspace(-2, 4, Nl)
            psi= np.linspace(np.arctan(0.5), np.pi/4, Npsi)
            u  = np.linspace(0.0, 1-1e-3, Nu)
            lg, psig, ug = np.meshgrid(l, psi, u, indexing='ij')
            mug = 0.5/np.tan(psig) + ug*(1-0.5/np.tan(psig))
        elif self.domain == 'domain2':
            l  = np.logspace(-2, 4, Nl)
            psi= np.linspace(1e-5, np.pi/4, Npsi)
            u  = np.linspace(0.0, 1, Nu)
            lg, psig, ug = np.meshgrid(l, psi, u, indexing='ij')
            mug = -1 + ug*(0.5*np.tan(psig) + 1)
        elif self.domain == 'domain3':
            l  = np.logspace(-2, 4, Nl)
            psi= np.linspace(1e-5, np.pi/4, Npsi)
            u  = np.linspace(0.0, 1, Nu)
            lg, psig, ug = np.meshgrid(l, psi, u, indexing='ij')
            up = np.min([0.5/np.tan(psig), np.ones(psig.shape)], axis=0)
            mug = 0.5*np.tan(psig) + ug*(up - 0.5*np.tan(psig))

        # Compute bispectrum on grids
        bl = self.bl_direct(lg, psig, mug, targtype='lpsimu')

        # Make Interpolation
        self.log_bl_interp = rgi((np.log(l), psi, u), np.log(bl), bounds_error=False, fill_value=None)

    def bl_interp(self, targ1, targ2, targ3, targtype='l1l2l3'):
        if targtype == 'l1l2l3':
            l, psi, mu = utils.l1l2l3_to_lpsimu_sorted(targ1, targ2, targ3, self.domain)
        elif targtype == 'lpsimu':
            l, psi, mu = utils.lpsimu_to_lpsimu_sorted(targ1, targ2, targ3, self.domain)

        if self.domain == 'domain1':
            u = (mu-0.5/np.tan(psi))/(1.0-0.5/np.tan(psi))
        elif self.domain == 'domain2':
            u = (mu+1)/(np.tan(psi)+1)
        elif self.domain == 'domain3':
            up = np.min([0.5/np.tan(psi), np.ones(psi.shape)], axis=0)
            u = (mu-0.5*np.tan(psi))/(up-0.5*np.tan(psi))

        log_bl = self.log_bl_interp((np.log(l), psi, u))
        bl = np.exp(log_bl)

        return bl

    def test(self, targ1, targ2, targ3, targtype='l1l2l3'):
        if targtype == 'l1l2l3':
            l, psi, mu = utils.l1l2l3_to_lpsimu_sorted(targ1, targ2, targ3, self.domain)
        elif targtype == 'lpsimu':
            l, psi, mu = utils.lpsimu_to_lpsimu_sorted(targ1, targ2, targ3, self.domain)

        return psi, mu
