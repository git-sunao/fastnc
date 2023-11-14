import matplotlib.pyplot as plt
import numpy as np
from . import trigutils
from scipy.special import sici

# plotter for Gamma0
def rad2min(v):
    return np.rad2deg(v)*60.0

def log_imshow(ax, z, extent, cmap='bwr', add_cbar=True, vmax=None):
    def cbfmt(x, pos):
        a, b = '{:.1e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    if vmax is None:
        vmax = np.max(np.abs(z))
    im = ax.imshow(z, vmin=-vmax, vmax=vmax, extent=extent, cmap=cmap)
    if add_cbar:
        plt.colorbar(im, ax=ax, format=cbfmt)
    ax.invert_yaxis()
    return vmax

def set_minute_extent(ax, x1, x2, xlabel=r'$x_1~[{\rm arcmin}]$', ylabel=r'$x_2~[{\rm arcmin}]$'):
    def helper(v, pos):
        return r'$10^{%d}$' % (v)

    ax.xaxis.set_major_formatter(helper)
    ax.yaxis.set_major_formatter(helper)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    logx1 = np.log10(rad2min(x1))
    logx2 = np.log10(rad2min(x2))

    extent = [logx1.min(), logx1.max(), logx2.max(), logx2.min()]
    return extent

def imshow_g0(x1, x2, g0, suptitle=None, xlabel=r'$x_1$', ylabel=r'$x_2$'):
    xx1, xx2 = np.meshgrid(x1, x2)
    z = g0*np.sqrt(xx1*xx2)

    fig, axes = plt.subplots(1,2, figsize=(8,3))
    plt.subplots_adjust(wspace=0.6)
    fig.suptitle(suptitle)

    axes[0].set_title('Real')
    extent = set_minute_extent(axes[0], x1, x2)
    im = log_imshow(axes[0], z.real, extent=extent)

    axes[1].set_title('Imag')
    extent = set_minute_extent(axes[1], x1, x2)
    im = log_imshow(axes[1], z.imag, extent=extent)
    return fig, axes

# Toy bispectrum
def rho_NFW(k, rs):
    y = k*rs
    si, ci = sici(y)
    return -np.cos(y)*ci + 0.5*np.sin(y)*(np.pi-2*si)

def bispectrum_NFW(l, psi, mu, rs_arcmin=10.0):
    l1, l2, l3 = trigutils.lpsimu_to_l1l2l3(l, psi, mu)
    bl = 1
    for i, _l in enumerate([l1, l2, l3]):
        # assume rs = 10 arcmin on sky
        rs = np.deg2rad(rs_arcmin/60.0)
        bl *= rho_NFW(_l, rs)
    return bl