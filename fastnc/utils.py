import numpy as np

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