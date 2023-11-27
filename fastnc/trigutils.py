"""
This module provides the functions to manipulate the triangle configuration.
Triangle can be specified by several different ways, and this module provides
the conversion between them.

1. `x1x2x3` -- (x1, x2, x3)
    Side lengths of triangle. This specification of triangle 
    is *unaware* of the triangle orientation.
2. `xpsimu` -- (x, psi, mu)
    With given side lengths of triangle, x1, x2, x3,
        x1 = x * cos(psi), 
        x2 = x * sin(psi), 
        x3 = x * sqrt(1 - sin(2*psi)*mu)
    and inversely
        x   = sqrt(x1^2 + x2^2), 
        psi = arctan(y=x2, x=x1), 
        mu  = (cosine of inner anglebetween x1 and x2)
    This specification of triangle is *unaware* of the triangle orientation.
3. `x1x2phi3` -- (x1, x2, phi3)
    With given side lengths of triangle, x1, x2, x3,
        phi3 = arccos( (x1^2 + x2^2 - x3^2)/2/x1/x2 ),
    i.e. phi3 is the angle between x1 and x2 sides.
    This specification of triangle is *unaware* of the triangle orientation.
4. `ruv` -- (r, u, v)
    The convention of `treecorr`. With given side lengths of triangle, x1>x2>x3,
        r = x2, 
        u = x3/x2, 
        v = \pm (x1-x2)/x3
    where the sign of v is determined by the orientation of triangle: 
    if the sides x1, x2, x3 are anti-clockwise, v is positive, otherwise negative. 
    This specification of triangle is *aware* of the triangle orientation.
5. `x1x2dvphi` -- (x1, x2, dvphi)
    The convention of `fastnc`. With given side lengths of triangle, x1>x2>x3,
        dvphi = phi3 - pi if v >= 0 else pi - phi3
    Thus dvphi is the oriented outer angle of triangle.
    This specification of triangle is *aware* of the triangle orientation.

Note that the conversion between the orientation-unaware specifications can be ambiguous.

Author: Sunao Sugiyama
Last edit: 2023/11/27
"""

import numpy as np

##############
def is_cyclic_permutation(x):
    N = len(x)
    
    for i in range(N):
        cyclic_permutation = np.roll(np.arange(N), i)
        if np.array_equal(x, cyclic_permutation):
            return True
    
    return False

def ruv_to_x1x2x3(r, u, v):
    x1 = r*(1+u*np.abs(v))
    x2 = r
    x3 = r*u
    return x1, x2, x3

def x1x2x3_to_ruv(x1, x2, x3):
    """
    x1, x2, x3 are clockwise side lengths of triangle
    """
    # check the (d1 > d2 > d3) triangle is clockwise or not
    idx = np.argsort([x1, x2, x3], axis=0).T
    clk = [is_cyclic_permutation(_idx) for _idx in idx]
    sign = np.ones_like(clk, dtype=int)
    sign[np.logical_not(clk)] = -1
    # Get sorted side length
    d3, d2, d1 = np.sort([x1,x2,x3], axis=0)
    r = d2
    u = d3/d2
    v = sign*(d1-d2)/d3

    return r, u, v

def ruv_to_x1x2dvphi(r, u, v):
    x1, x2, x3 = (1+u*np.abs(v)), 1., u
    phi3 = np.arccos( (x1**2+x2**2-x3**2)/2/x1/x2 )
    if v >= 0:
        dvphi = phi3 - np.pi
    else:
        dvphi = np.pi - phi3
    x1 *= r
    x2 *= r
    return x1, x2, dvphi

def ruv_to_x2x3dvphi(r, u, v):
    x1, x2, x3 = (1+u*np.abs(v)), 1., u
    phi1 = np.arccos( (x2**2+x3**2-x1**2)/2/x2/x3 )
    if v >= 0:
        dvphi = phi1 - np.pi
    else:
        dvphi = np.pi - phi1
    x2 *= r
    x3 *= r
    return x2, x3, dvphi

def ruv_to_x3x1dvphi(r, u, v):
    x1, x2, x3 = (1+u*np.abs(v)), 1., u
    phi2 = np.arccos( (x3**2+x1**2-x2**2)/2/x3/x1 )
    if v >= 0:
        dvphi = phi2 - np.pi
    else:
        dvphi = np.pi - phi2
    x3 *= r
    x1 *= r
    return x3, x1, dvphi

def x1x2dvphi_to_x1x2x3(x1, x2, dvphi):
    x3 = np.sqrt(x1**2 + x2**2 + 2*x1*x2*np.cos(dvphi))
    return x1, x2, x3

def xpsimu_to_x1x2x3(x, psi, mu):
    x1 = x * np.cos(psi)
    x2 = x * np.sin(psi)
    x3 = x * np.sqrt(1 - np.sin(2*psi)*mu)
    return x1, x2, x3