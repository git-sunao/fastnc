from astropy.cosmology import wCDM
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

class cosmology(wCDM):
    def __init__(self, zmin, zmax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_interpolator(zmin, zmax)

    def init_interpolator(self, zmin, zmax):
        # redshift
        z   = np.linspace(zmin, zmax, 500)
        # comoving distance in unit of Mpc/h
        h   = self.h
        chi = self.comoving_distance(z).value*h
        fK  = self.comoving_transverse_distance(z).value*h
        self.z2chi = ius(z, chi)
        self.z2fK  = ius(z, fK)
        self.chi2z = ius(chi, z)
