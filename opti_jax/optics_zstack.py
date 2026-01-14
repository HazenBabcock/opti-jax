#!/usr/bin/env python
"""
Optics for simulating / solving from a (bright-field) Z stack.

Hazen 2026.01
"""


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import opti_jax.optics as optics


class OpticsZStack(optics.OpticsBF):
    """
    Z (focus) stack.
    """
    def __init__(self, ni = 20, nj = 50, **kwds):
        """
        Default is 1000 iterations for solving (ni * nj).
        """
        super().__init__(**kwds)

        self.bf_pat = None
        self.ni = ni
        self.nj = nj

        
    def compute_loss_tv_order1(self, x, Y, zvals, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.z_stack(x, zvals)
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order1(x)*lval
        return loss


    def compute_loss_tv_order2(self, x, Y, zvals, lval):
        """
        Total variation loss function, first order.
        """
        yPred = self.z_stack(x, zvals)
        loss = jnp.mean(optax.l2_loss(yPred, Y)) + self.tv_smoothness_order2(x)*lval
        return loss


    def make_bf_pattern(self, maxNA, ik0, ik1):
        """
        Approximate uniform illumination over an aperture with multiple point sources.
        
        ik0 and ik1 are expected to be integer arrays.
        """
        self.bf_pat = super().make_bf_pattern(maxNA, ik0, ik1)
        return self.bf_pat
    
    
    def tv_solve(self, Y, illm, lval = 1.0e-5, learningRate = 1.0e-1, order = 2, verbose = True):
        """
        Defaults tuned for a bright field focus stack.
        """
        return super().tv_solve(Y, illm, lval = lval, learningRate = learningRate, order = order, verbose = verbose)


#    def z_stack(self, xrc, zvals):
#        """
#        Return images for each of the illumination z values.
#        """
#        rxy = jnp.array(self.bf_pat)
#        
#        tmp = []
#        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
#        for zv in zvals:
#            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv) * self.mask
#            pim = jnp.zeros_like(xrcFT)
#            for rx, ry in rxy:
#                pim += jnp.roll(zshift, (-rx,-ry), (0,1)) * xrcFT
#            tmp.append(self.intensity(self.from_fourier(pim)/float(len(rxy))))
#        return jnp.array(tmp)


    def z_stack(self, xrc, zvals):
        """
        Return images for each of the illumination z values.
        """
        tmp = []
        xrcFT = self.to_fourier(self.illuminate(xrc, 0.0, 0.0))
        for zv in zvals:
            zshift = jnp.exp(1j * 2.0 * jnp.pi * self.kz * zv)
            pim = jnp.zeros(xrcFT.shape)
            for rx, ry in jnp.array(self.bf_pat):
                pim = pim + self.intensity(self.from_fourier(jnp.roll(xrcFT, (rx,ry), (0,1)) * zshift * self.mask))
            tmp.append(pim/float(len(self.bf_pat)))
        return jnp.array(tmp)
