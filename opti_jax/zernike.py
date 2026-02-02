#!/usr/bin/env python
"""
Zernike polynomial creation.

This requires the 'zernike' package.
https://github.com/jacopoantonello/zernike

Hazen 2026.02
"""

import numpy as np
import zernike


def zern_poly(oz, zn):
    """
    Return zernike polynomial zn that matches the aperture of the Optics class oz.

    zn is the Noll index?
    """
    x0 = oz.mask*(oz.k0/oz.kmax)
    x1 = oz.mask*(oz.k1/oz.kmax)
    
    cart = zernike.RZern(6)
    cart.make_cart_grid(x0, x1)
    
    c = np.zeros(cart.nk)
    c *= 0.0
    c[zn] = 1.0
    return cart.eval_grid(c, matrix=True)
