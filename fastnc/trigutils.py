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
    # # 1
    phi3 = np.arccos( (1 + (1+u*np.abs(v))**2 - u**2)/2/(1+u*np.abs(v)) )
    if v >= 0:
        dvphi = phi3 - np.pi
    else:
        dvphi = np.pi - phi3
    x1 = r*(1+u*np.abs(v))
    x2 = r

    # 2
    # phi3 = np.arccos( (1 + u**2 - (1+u*np.abs(v))**2)/2/u )
    # if v >= 0:
    #     dvphi = phi3 - np.pi
    # else:
    #     dvphi = np.pi - phi3
    # x1 = r
    # x2 = r*u

    # 3
    # phi3 = np.arccos( (u**2 + (1+u*np.abs(v))**2 - 1)/2/u/(1+u*np.abs(v)) )
    # if v >= 0:
    #     dvphi = phi3 - np.pi
    # else:
    #     dvphi = np.pi - phi3
    # x1 = r*(1+u*np.abs(v))
    # x2 = r*u

    return x1, x2, dvphi

##############


def l1l2l3_to_lminlmidlmax(l1, l2, l3):
    lmin, lmid, lmax = np.sort([l1, l2, l3], axis=0)
    return lmin, lmid, lmax

def l1l2l3_to_lpsimu(l1, l2, l3):
    """
    l1 = l * cos(psi)
    l2 = l * sin(psi)
    l3 = l * sqrt(1 - sin(2*psi)*mu)

    l    = sqrt(l1^2 + l2^2)
    psi  = arctan(y=l2, x=l1)
    mu   = (1- (l3/l)^2) / sin(2*psi)
    """
    l   = np.sqrt(l1**2 + l2**2)
    psi = np.arctan2(l2, l1)
    mu  = (1 - (l3/l)**2) / np.sin(2*psi)
    return l, psi, mu

def lpsimu_to_l1l2l3(l, psi, mu):
    """
    l1 = l * cos(psi)
    l2 = l * sin(psi)
    l3 = l * sqrt(1 - sin(2*psi)*mu)
    """
    l1 = l*np.cos(psi)
    l2 = l*np.sin(psi)
    l3 = l*np.sqrt(1-np.sin(2*psi)*mu)
    return l1, l2, l3

# Interface
def get_l1l2l3(targ1, targ2, targ3, targtype):
    if targtype == 'l1l2l3':
        return targ1, targ2, targ3
    elif targtype == 'lpsimu':
        return lpsimu_to_l1l2l3(targ1, targ2, targ3)
    else:
        print('Error: Expected targtype is one of "l1l2l3" or "lpsimu": {} is not expected'.format(targtype))
        return targ1, targ2, targ3

def get_lpsimu(targ1, targ2, targ3, targtype):
    if targtype == 'l1l2l3':
        return l1l2l3_to_lpsimu(targ1, targ2, targ3)
    elif targtype == 'lpsimu':
        return targ1, targ2, targ3
    else:
        print('Error: Expected targtype is one of "l1l2l3" or "lpsimu": {} is not expected'.format(targtype))
        return targ1, targ2, targ3

def lpsimu_to_lpsimu_sorted(l, psi, mu, domain):
    l1, l2, l3 = lpsimu_to_l1l2l3(l, psi, mu)
    lmin, lmid, lmax = l1l2l3_to_lminlmidlmax(l1, l2, l3)
    if domain == 'domain1':
        l, psi, mu = l1l2l3_to_lpsimu(lmax, lmid, lmin)
    elif domain == 'domain2':
        l, psi, mu = l1l2l3_to_lpsimu(lmid, lmin, lmax)
    elif domain == 'domain3':
        l, psi, mu = l1l2l3_to_lpsimu(lmax, lmin, lmid)
    return l, psi, mu

def l1l2l3_to_lpsimu_sorted(l1, l2, l3, domain):
    lmin, lmid, lmax = l1l2l3_to_lminlmidlmax(l1, l2, l3)
    l, psi, mu = l1l2l3_to_lpsimu(lmax, lmid, lmin)
    return l, psi, mu